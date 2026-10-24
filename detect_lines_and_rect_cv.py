import math
import os
from collections import namedtuple

import cv2
import numpy as np

# 一條線的結果：兩端點 + 品質指標
Line = namedtuple('Line', 'x1 y1 x2 y2 length fill dev')

SIDES = ('left', 'right', 'top', 'bottom')


# ------------------------------------------------ 長線紀錄：後續找矩形要用的幾何量
class EdgeLine(namedtuple('EdgeLine', 'x1 y1 x2 y2 length fill dev axis angle side')):
    """一條「夠長的線」記下來的樣子：detect_lines 的 Line 再補上找矩形要用的東西。

    axis  'v' / 'h'：這條線比較接近鉛直還是水平（看端點誰的落差大）。
    angle 對該軸的傾角（度）：'v' 從鉛直線量起（往下走時往右偏為正），
          'h' 從水平線量起（往右走時往下偏為正）。給的是帶正負號的量，
          因為找四角時四邊的傾角要互相對得起來，絕對值會分不出左右。
    side  當初是哪一側掃出來的（left/right/top/bottom），之後分左右上下邊界用。
    端點已經排好順序：'v' 是 (上端, 下端)，'h' 是 (左端, 右端)。
    """

    __slots__ = ()

    @property
    def p0(self):
        return np.array((self.x1, self.y1), np.float64)

    @property
    def p1(self):
        return np.array((self.x2, self.y2), np.float64)

    @property
    def cx(self):
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self):
        return (self.y1 + self.y2) / 2.0

    @property
    def unit(self):
        """從 p0 指向 p1 的單位向量：'v' 大致朝下、'h' 大致朝右。"""
        v = self.p1 - self.p0
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            return np.zeros(2)
        return v / n

    @property
    def normal(self):
        """垂直於線方向的單位向量。"""
        u = self.unit
        return np.array([-u[1], u[0]])

    def padded(self, pad):
        """兩端各往外伸 pad 像素，回傳新的 (p0, p1)。

        不是為了讓交叉點算得出來 —— 兩條無限長直線的交點跟截多長完全無關。
        延伸是為了量「交叉點離這條線多遠」：線斷掉時角點本來就會落在斷口外
        一點點（反光、折角、貼邊被切掉），偏一點算正常；偏太遠就代表這條線
        根本不是同一個矩形的邊，再怎麼延伸也接不回來。
        """
        u = self.unit * float(pad)
        return self.p0 - u, self.p1 + u


# ---------------------------------------------------------------- 邊緣點
def _edge_pixels(img, kind='gradient', lo=40, hi=120, pct=88, bridge=1):
    """影像裡所有可用來生長的點，回傳 [(x, y), ...]。

    kind='gradient' 用梯度幅值取第 pct 百分位以上當點。
        實測在低對比的收據上比 Canny 好：桌面上半段是平滑漸層，Canny 整個沒反應，
        收據右緣因此「不存在」；梯度有反應，右緣才找得回來。
    kind='canny'    用 Canny，點比較乾淨，但弱對比的邊會整條消失。
    bridge > 0 會把點膨脹一點點，讓生長可以跨過自然斷掉的小缺口。
    """
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    if kind == 'canny':
        mask = cv2.Canny(gray, lo, hi)
    elif kind == 'gradient':
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.normalize(cv2.magnitude(gx, gy), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        mask = (mag >= np.percentile(mag, pct)).astype(np.uint8) * 255
    else:
        raise ValueError("kind 只能是 'canny' 或 'gradient'")
    if bridge > 0:
        mask = cv2.dilate(mask, np.ones((2 * bridge + 1, 2 * bridge + 1), np.uint8))
    ys, xs = np.nonzero(mask)
    return list(zip(xs.tolist(), ys.tolist()))


def _build_grid(points, cell):
    """把點分格存起來，找鄰居時只看自己跟周圍 8 格，不用掃全圖。"""
    grid = {}
    for i, (x, y) in enumerate(points):
        grid.setdefault((x // cell, y // cell), []).append(i)
    return grid


def _neighbors(i, points, grid, cell, max_dist):
    x, y = points[i]
    cx, cy = x // cell, y // cell
    r2 = max_dist * max_dist
    out = []
    for gy in (cy - 1, cy, cy + 1):
        for gx in (cx - 1, cx, cx + 1):
            for j in grid.get((gx, gy), ()):
                dx, dy = points[j][0] - x, points[j][1] - y
                if (dx or dy) and dx * dx + dy * dy <= r2:
                    out.append(j)
    return out


# ---------------------------------------------------------------- 生長：像神經一樣連點
def _trace(seed, points, grid, cell, max_dist, cos_turn, init_dir, max_steps, corridor=None):
    """從種子點沿邊緣往外長。

    每個點都帶著「我從哪個方向走過來」，下一個點跟這個方向夾角夠小才接得上，
    長出來才是一條線而不是整片邊緣攤開。回傳所有連上的點。

    corridor = (中心點, 垂直向量, 方向向量, 寬度, 錐度)：
        給了就只准在這條線的走廊裡長、而且只能往同一個方向走。
        走廊是錐形：允許的偏移會隨離種子點的距離線性放寬（寬度 + 錐度 × 距離），
        因為真實的紙邊常常是「直線在中間鼓出去一個折角」，固定寬度會在折角處斷掉。
        完全沒給會從文件邊界一路鑽進文件裡印的字（實測鑽了 2000 多個點進去）。
    """
    sign = 1.0
    if corridor is not None:
        _, _, direction, _, _ = corridor
        sign = 1.0 if (init_dir[0] * direction[0] + init_dir[1] * direction[1]) >= 0 else -1.0

    chain, visited = [seed], {seed}
    frontier = [(seed, init_dir)]
    while frontier and len(chain) < max_steps:
        i, (ux, uy) = frontier.pop()
        px, py = points[i]
        for j in _neighbors(i, points, grid, cell, max_dist):
            if j in visited:
                continue
            vx, vy = points[j][0] - px, points[j][1] - py
            d = math.hypot(vx, vy)
            if d < 1e-6 or (vx * ux + vy * uy) / d < cos_turn:  # 轉角太大就不接
                continue
            if corridor is not None:
                center, normal, direction, width, bow = corridor
                ox, oy = points[j][0] - center[0], points[j][1] - center[1]
                progress = ox * direction[0] + oy * direction[1]
                if abs(ox * normal[0] + oy * normal[1]) > width + bow * abs(progress):
                    continue  # 超出走廊
                if progress * sign < 0:  # 回頭路了
                    continue
            visited.add(j)
            chain.append(j)
            frontier.append((j, (vx / d, vy / d)))
    return chain


def _group_seeds(seeds, tol):
    """種子點依掃描軸的位置分群：連續的掃描線會對到同一條邊界上。"""
    groups = []
    for value, point in sorted(seeds):
        if groups and abs(value - groups[-1][-1][0]) <= tol:
            groups[-1].append((value, point))
        else:
            groups.append([(value, point)])
    return groups


# ---------------------------------------------------------------- 驗證：像不像一條線
def _fit_line(pts):
    """最小平方直線擬合，回傳 (中心點, 方向向量, 各點到線的垂直距離)。"""
    xy = np.asarray(pts, np.float64)
    center = xy.mean(axis=0)
    d = xy - center
    _, _, vt = np.linalg.svd(d, full_matrices=False)
    direction = vt[0]
    normal = np.array([-direction[1], direction[0]])
    return center, direction, d @ normal


def _trim_outliers(pts, k=2.5, min_keep=0.4, rounds=2):
    """擬合完把離群點剔掉再重新擬合，回傳跟 _fit_line 一樣的三個值。

    種子群是「沿掃描軸的位置相近」分出來的，只看 x（左右掃）或 y（上下掃），
    所以兩條互不相干、只是位置接近的邊界會混進同一群：
    3.jpg 的右邊界在 x≈315，旁邊的桌沿在 x≈330，同一群 645 個種子，
    混在一起擬合出來的線兩邊都不像，生的走廊整個偏掉，右邊界就長不出來
    （實測 chain 245 點、長度 48px；剔掉離群點後 chain 2427 點、長度 313px）。
    """
    xy = np.asarray(pts, np.float64)
    for _ in range(rounds):
        center, direction, perp = _fit_line(xy)
        keep = np.abs(perp) <= max(4.0, k * float(np.median(np.abs(perp))))
        if keep.sum() < max(10, min_keep * len(xy)):  # 剔太多就當沒得剔
            break
        xy = xy[keep]
    return _fit_line(xy)


def _make_line(chain_pts):
    """把長出來的點收斂成 (起點, 終點, 長度, 最大垂直偏移)。"""
    center, direction, perp = _fit_line(chain_pts)
    half = (chain_pts - center) @ direction
    p0 = center + direction * half.min()
    p1 = center + direction * half.max()
    return p0, p1, float(half.max() - half.min()), float(np.abs(perp).max())


def _line_quality(chain_pts, center, direction, reach, samples=50):
    """沿著擬合線等距取樣 samples 個點，判斷這團點到底像不像一條線。

    coverage  ：多少比例的取樣點附近真的連著點（「連起來的點夠不夠多」）
    angle_std ：相鄰取樣點之間角度的標準差，越小越直（度）
    """
    half = (chain_pts - center) @ direction
    ts = np.linspace(half.min(), half.max(), samples)
    spots = center[None, :] + direction[None, :] * ts[:, None]                # (samples, 2)
    gaps = np.linalg.norm(chain_pts[:, None, :] - spots[None, :, :], axis=2)  # (N, samples)
    coverage = float((gaps.min(axis=0) <= reach).mean())

    steps = np.diff(chain_pts[gaps.argmin(axis=0)], axis=0)  # (samples-1, 2)
    lens = np.linalg.norm(steps, axis=1)
    keep = lens > 1e-6
    if keep.sum() < samples * 0.5:  # 取樣點大量重合，這團點沒有連續性
        return coverage, 180.0
    ang = np.arctan2(steps[keep, 1], steps[keep, 0])
    rel = np.unwrap(ang - np.arctan2(direction[1], direction[0]))
    rel = (rel + np.pi / 2) % np.pi - np.pi / 2  # 折疊到 ±90°
    return coverage, float(np.degrees(np.std(rel)))


def _dedupe(lines, tol):
    """同一條線可能被多個種子群各長一次，只留最長的那條。"""
    kept = []
    for line in sorted(lines, key=lambda l: -l.length):
        mx, my = (line.x1 + line.x2) / 2, (line.y1 + line.y2) / 2
        if any(abs(mx - (k.x1 + k.x2) / 2) < tol and abs(my - (k.y1 + k.y2) / 2) < tol for k in kept):
            continue
        kept.append(line)
    return kept


# ---------------------------------------------------------------- 主流程
def detect_lines(img, direction='vertical', kind='gradient', pct=88, lo=40, hi=120, bridge=1,
                 radius=3, max_dist=4, max_turn=35.0,
                 min_len_v=0.30, min_len_h=0.20, min_seed_ratio=0.05, fill_ratio=0.60,
                 straight_tol=None, corridor=None, bow=0.03, samples=50,
                 max_angle_std=10.0, max_steps=8000):
    """從影像某一側開始：每條掃描線找出該側第一個邊緣點當種子，
    同一條邊界上的種子點分群，群夠大才值得往深裡長，
    長完再檢查是不是夠直、夠長、夠連續的一條線。

    direction : 'vertical'   每欄由上往下找（抓文件的左右邊界）
                'horizontal' 每列由左往右找（抓文件的上/下邊界）
                'left' 'right' 'top' 'bottom' 指定從哪一側掃
                'auto'       四側都試
    min_len_v  ：left/right 的最短長度，佔影像「高度」的比例
    min_len_h  ：top/bottom 的最短長度，佔影像「寬度」的比例
    min_seed_ratio：多少比例的掃描線要對到同一條邊界才算「點連得起來」
    fill_ratio   ：線段上多少比例的長度真的被點蓋到
    回傳 Line 的 list，由長到短。
    """
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    points = _edge_pixels(gray, kind, lo, hi, pct, bridge)
    if len(points) < 10:
        return []

    cell = max(radius, 2)
    grid = _build_grid(points, cell)
    cos_turn = math.cos(math.radians(max_turn))
    straight_tol = straight_tol if straight_tol is not None else 0.02 * min(h, w) + 2
    # 走廊寬度＝生長可以偏離種子線多遠。給太寬不是比較保險，是會出事：
    # 線長到紙的角上時會順著旁邊那條邊跑掉。3.jpg 的左邊界就這樣鑽進下邊界，
    # 尾巴整段偏離 15px，明明是一條 533px 的好邊界，卻因為 dev 超標被丟掉
    # （走廊 10.3px → dev 15.2 失敗，收到 4.2px → dev 8.8 通過）。
    # 種子群剔過離群點之後擬合線本來就 很準，1% 的寬度綽綽有餘。
    corridor_w = corridor if corridor is not None else max(4.0, 0.01 * min(h, w))
    # 紙邊允許的彎曲量：佔線長的比例。3% 的話 530px 的邊可以鼓出約 16px，
    # 足以蓋住收據中段那個折角，又不至於讓生長跑到紙內的字上。
    max_dev = max(straight_tol, bow * min(h, w))
    # 生長方向要「沿著邊界線」而不是穿進文件：從左/右側掃到的邊界是鉛直線，
    # 從上/下側掃到的是水平線。線有兩端，兩個方向都要長，否則只長到一半。
    axis_dirs = {'left': (0.0, 1.0), 'right': (0.0, 1.0),
                 'top': (1.0, 0.0), 'bottom': (1.0, 0.0)}

    if direction == 'auto':
        sides = SIDES
    elif direction in ('vertical', 'horizontal'):
        sides = ('top', 'bottom') if direction == 'vertical' else ('left', 'right')
    elif direction in SIDES:
        sides = (direction,)
    else:
        raise ValueError("direction 只能是 'vertical' 'horizontal' 'left' 'right' 'top' 'bottom' 'auto'")

    is_edge = np.zeros((h, w), bool)
    edge_xy = np.asarray(points)
    is_edge[edge_xy[:, 1], edge_xy[:, 0]] = True
    ys_all, xs_all = np.nonzero(is_edge)

    lines = []
    for side in sides:
        # 每條掃描線找該側第一個邊緣點
        if side in ('left', 'right'):
            seeds = []
            for y in np.unique(ys_all):
                xs = xs_all[ys_all == y]
                x = int(xs.min()) if side == 'left' else int(xs.max())
                seeds.append((x, (x, int(y))))
            scan_lines, min_len = h, min_len_v * h   # 鉛直邊界跟影像高度比
        else:
            seeds = []
            for x in np.unique(xs_all):
                ys = ys_all[xs_all == x]
                y = int(ys.min()) if side == 'top' else int(ys.max())
                seeds.append((y, (int(x), y)))
            scan_lines, min_len = w, min_len_h * w   # 水平邊界跟影像寬度比
        min_seeds = int(min_seed_ratio * scan_lines)

        for group in _group_seeds(seeds, max_dist):
            if len(group) < min_seeds:  # 對不起來，掃描線太少 → 不成條線
                continue
            seed_pts = np.asarray([p for _, p in group], np.float64)
            g_center, g_dir, _ = _trim_outliers(seed_pts)
            g_normal = np.array([-g_dir[1], g_dir[0]])
            corridor = (g_center, g_normal, g_dir, corridor_w, bow)

            cx = int(round(g_center[0]))
            cy = int(round(g_center[1]))
            seed = min(range(len(points)),
                       key=lambda i: abs(points[i][0] - cx) + abs(points[i][1] - cy))
            ux, uy = axis_dirs[side]
            chain = _trace(seed, points, grid, cell, max_dist, cos_turn, (ux, uy), max_steps, corridor)
            chain = list({*chain, *_trace(seed, points, grid, cell, max_dist, cos_turn,
                                          (-ux, -uy), max_steps, corridor)})
            if len(chain) < 20:
                continue
            chain_pts = np.asarray([points[i] for i in chain], np.float64)
            p0, p1, length, dev = _make_line(chain_pts)
            if length < min_len or dev > max_dev:
                continue
            center, dirv, _ = _fit_line(chain_pts)
            coverage, angle_std = _line_quality(chain_pts, center, dirv, max_dev, samples)
            if coverage < fill_ratio or angle_std > max_angle_std:
                continue
            lines.append(Line(int(p0[0]), int(p0[1]), int(p1[0]), int(p1[1]),
                              length, coverage, angle_std))

    return _dedupe(lines, max_dev)


# ------------------------------------------------ 記錄：留下夠長的線給後面用
def _make_edge_line(line, side):
    """把 detect_lines 的 Line 升級成 EdgeLine：分鉛直/水平、算傾角、排好端點順序。"""
    a, b = (line.x1, line.y1), (line.x2, line.y2)
    if a == b:
        return None  # 零長度，沒有方向可言
    if abs(b[1] - a[1]) >= abs(b[0] - a[0]):          # y 落差比較大 → 比較接近鉛直
        a, b = sorted((a, b), key=lambda p: p[1])     # a 在上、b 在下
        angle = math.degrees(math.atan2(b[0] - a[0], b[1] - a[1]))
        axis = 'v'
    else:
        a, b = sorted((a, b), key=lambda p: p[0])     # a 在左、b 在右
        angle = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
        axis = 'h'
    return EdgeLine(a[0], a[1], b[0], b[1], line.length, line.fill, line.dev, axis, angle, side)


def _drop_dupes(records, tol):
    """同一條線可能被左右上下各掃到一次，只留最長的那條。

    detect_lines 只在「單一側」的結果裡去重，跨側的重複要在這裡再清一次，
    否則後面算交叉點時同一條邊會出現兩份，四角怎麼配都會多出候選。
    """
    kept = []
    for rec in sorted(records, key=lambda r: -r.length):
        if any(rec.axis == k.axis and abs(rec.angle - k.angle) < 15.0
               and abs(rec.cx - k.cx) < tol and abs(rec.cy - k.cy) < tol for k in kept):
            continue
        kept.append(rec)
    return kept


def record_long_lines(img, pct_v=0.30, pct_h=0.20, **kwargs):
    """把四側掃到的線裡「夠長」的全部記錄下來，供後續找矩形用。

    pct_v ：鉛直線要超過影像「高度」的幾成才算數
    pct_h ：水平線要超過影像「寬度」的幾成才算數
    其他參數（kind、pct、bridge、bow…）原樣傳給 detect_lines。

    回傳依長度由長到短的 EdgeLine list，鉛直水平混在一起，要分流的話
    自己用 e.axis == 'v' 挑。線斷成兩截時會各自算一條，
    之後要接起來再說（把共線且相鄰的片段併起來）。
    """
    h, w = img.shape[:2]
    # 長度門檻一律由這裡決定，不要讓 detect_lines 的預設值（0.30/0.20）
    # 偷偷先砍掉一批「比預設短、但符合這裡 pct_v/pct_h」的線。
    kwargs.setdefault('min_len_v', min(pct_v, 0.05))
    kwargs.setdefault('min_len_h', min(pct_h, 0.05))

    records = []
    for side in SIDES:
        for line in detect_lines(img, direction=side, **kwargs):
            rec = _make_edge_line(line, side)
            if rec is None:
                continue
            need = pct_h * w if rec.axis == 'h' else pct_v * h
            if rec.length < need:
                continue
            records.append(rec)

    # 去重尺度跟 detect_lines 內部一致：同一條線在兩側都被掃到時才不會留兩份
    tol = max(0.02 * min(h, w) + 2, kwargs.get('bow', 0.03) * min(h, w))
    return sorted(_drop_dupes(records, tol), key=lambda r: -r.length)


# ------------------------------------------------ 找矩形：兩條鉛直邊 × 兩條水平邊
def _line_cross(v, h):
    """兩條線當成無限長直線的交點；近乎平行回傳 None。

    注意用的是無限長直線不是線段：線斷掉時角點本來就落在斷口外
    （3.jpg 的右下角在 (327.5, 579)，右邊界只偵測到 y=367），
    要拿偵測到的長度去卡交點，卡掉的就正是我們要的角。
    """
    a, b = v.p1 - v.p0, h.p1 - h.p0
    denom = a[0] * b[1] - a[1] * b[0]
    if abs(denom) < 1e-9:
        return None
    d = h.p0 - v.p0
    return v.p0 + ((d[0] * b[1] - d[1] * b[0]) / denom) * a


def _seg_dist(p, a, b):
    """點 p 到「線段」ab 的最短距離；p 在線段中間時不會掉到直線距離。"""
    ab = b - a
    L2 = float(ab @ ab)
    if L2 < 1e-9:
        return float(np.linalg.norm(p - a))
    t = min(1.0, max(0.0, float((p - a) @ ab) / L2))   # 夾住，落在外頭就卡在端點
    return float(np.linalg.norm(p - (a + t * ab)))


def _corner_angles(quad):
    """四個內角（度）。任一邊退化（長度 0）回傳 None。"""
    angs = []
    for i, p in enumerate(quad):
        u, v = quad[i - 1] - p, quad[(i + 1) % len(quad)] - p
        nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
        if nu < 1e-9 or nv < 1e-9:
            return None
        angs.append(math.degrees(math.acos(float(np.clip((u @ v) / (nu * nv), -1.0, 1.0)))))
    return angs


def _is_convex(quad):
    """四角照 TL→TR→BR→BL 排是不是凸的。兩條邊交叉會變蝴蝶形，要擋掉。"""
    signs = [(quad[(i + 1) % 4][0] - quad[i][0]) * (quad[(i + 2) % 4][1] - quad[(i + 1) % 4][1])
             - (quad[(i + 1) % 4][1] - quad[i][1]) * (quad[(i + 2) % 4][0] - quad[(i + 1) % 4][0])
             for i in range(4)]
    return all(s > 0 for s in signs) or all(s < 0 for s in signs)


def _corners(left, right, top, bottom):
    """兩條鉛直 × 兩條水平 → (左上, 右上, 右下, 左下)；有一邊平行就沒得交。"""
    quad = [_line_cross(left, top), _line_cross(right, top),
            _line_cross(right, bottom), _line_cross(left, bottom)]
    return None if any(p is None for p in quad) else quad


def _support(quad, sides, tol, far):
    """四角有多被偵測到的線段支撐（0~1）。

    一個角只要有一邊接得上就算數——另一邊可能正好斷在角上；
    四個角都離所有線段很遠的話，這幾條線根本不是同一個框。
    """
    (tl, tr, br, bl), (left, right, top, bottom) = quad, sides
    scores = []
    for p, v, h in ((tl, left, top), (tr, right, top), (br, right, bottom), (bl, left, bottom)):
        d = min(_seg_dist(p, v.p0, v.p1), _seg_dist(p, h.p0, h.p1))
        scores.append(1.0 if d <= tol else max(0.0, 1.0 - (d - tol) / max(far - tol, 1e-6)))
    return sum(scores) / len(scores)


def find_rect(edges, scale, min_area, max_tilt=20.0, min_side_ratio=0.60,
              max_corner_dev=25.0, support_tol=0.05, max_extend=0.25,
              min_support=0.5, min_side_cover=0.45):
    """從記錄到的長線裡挑出最像矩形的一組，回傳 (左上, 右上, 右下, 左下) 或 None。

    先用 max_tilt 濾掉太斜的線（3.jpg 右下角那條 -43° 的桌沿就是這樣被擋掉的，
    它長達 56% 寬度，光看長度會被當成水平邊），剩下鉛直線、水平線兩兩配對，
    交點就是候選四角，再逐一檢查是不是矩形。

    min_side_ratio ：對邊長度比的容許範圍（0.60~1.65），跟 main.py 拉平時用的是同一套，
                     刻意放得寬：拍下來的文件本來就會因透視變成梯形，
                     3.jpg 的收據上邊 152px、下邊 201px，比例 0.76，嚴格矩形會直接否決。
    support_tol    ：角點離線段多遠算「接得上」，佔畫面短邊的比例。
    max_extend     ：再遠就當這條線不是這條邊。
    min_side_cover ：每條邊線至少要覆蓋它那條邊的幾成，擋掉「拿一小段碎片外推出整條邊」。
    回傳的分數（面積 × 支撐度）只用來在多組候選裡排名，沒有另外回傳。
    """
    V = sorted((e for e in edges if e.axis == 'v' and abs(e.angle) <= max_tilt),
               key=lambda e: e.cx)          # 由左而右
    H = sorted((e for e in edges if e.axis == 'h' and abs(e.angle) <= max_tilt),
               key=lambda e: e.cy)          # 由上而下
    if len(V) < 2 or len(H) < 2:
        return None
    tol, far = support_tol * scale, max_extend * scale

    best, best_score = None, 0.0
    for i in range(len(V)):
        for j in range(i + 1, len(V)):
            for k in range(len(H)):
                for m in range(k + 1, len(H)):
                    left, right, top, bottom = V[i], V[j], H[k], H[m]
                    quad = _corners(left, right, top, bottom)
                    if quad is None or not _is_convex(quad):
                        continue
                    area = abs(cv2.contourArea(np.asarray(quad, np.float32)))
                    if area < min_area:
                        continue
                    angs = _corner_angles(quad)
                    if angs is None or max(abs(a - 90.0) for a in angs) > max_corner_dev:
                        continue
                    # 對邊長度比：梯形也放行，但差太多就不是一個框了
                    top_len = np.linalg.norm(quad[1] - quad[0])
                    bot_len = np.linalg.norm(quad[2] - quad[3])
                    left_len = np.linalg.norm(quad[3] - quad[0])
                    right_len = np.linalg.norm(quad[2] - quad[1])
                    if min(top_len / bot_len, bot_len / top_len,
                           left_len / right_len, right_len / left_len) < min_side_ratio:
                        continue
                    # 每條邊線都要真的覆蓋它那條邊的一段，不能整條邊都是外推的。
                    # 線太短也能通過上面所有檢查（角點照樣落在線上，支撐度很高），
                    # 但拼出來的框會從紙的邊界中間穿過去，把內容切掉：
                    # 1.jpg 最誇張的一次，左邊界只剩 125px、框的左邊有 548px，
                    # 覆蓋率 0.23，框的上緣直接橫切過「RESTORAN HASSANBISTRO」那行字。
                    # 真正合法的斷線還留得住：3.jpg 的右邊界只偵測到 313px / 522px = 0.60。
                    if min(left.length / left_len, right.length / right_len,
                           top.length / top_len, bottom.length / bot_len) < min_side_cover:
                        continue
                    sup = _support(quad, (left, right, top, bottom), tol, far)
                    if sup < min_support:
                        continue
                    score = area * sup
                    if score > best_score:
                        best, best_score = quad, score
    return best


def detect_rect(img, min_area_ratio=0.15, pct_v=0.30, pct_h=0.20, max_tilt=20.0,
                min_side_ratio=0.60, max_corner_dev=25.0, support_tol=0.05,
                max_extend=0.25, min_support=0.5, min_side_cover=0.45, **kwargs):
    """一張圖 → 文件的四個角 (左上, 右上, 右下, 左下)；不像矩形就回傳 None。

    pct_v / pct_h 與 kwargs（kind、pct、bow…）給 record_long_lines 偵測線，
    max_tilt 以下給 find_rect 驗證幾何。兩邊的門檻分開調，別互相綁死。
    """
    h, w = img.shape[:2]
    edges = record_long_lines(img, pct_v=pct_v, pct_h=pct_h, **kwargs)
    quad = find_rect(edges, scale=min(h, w), min_area=min_area_ratio * h * w,
                     max_tilt=max_tilt, min_side_ratio=min_side_ratio,
                     max_corner_dev=max_corner_dev, support_tol=support_tol,
                     max_extend=max_extend, min_support=min_support,
                     min_side_cover=min_side_cover)
    return None if quad is None else np.asarray(quad, np.float64)


# ---------------------------------------------------------------- 畫出來看
def draw_lines(img, lines, color=(0, 0, 255), thickness=2):
    out = img.copy()
    for line in lines:
        cv2.line(out, (line.x1, line.y1), (line.x2, line.y2), color, thickness)
        cv2.circle(out, (line.x1, line.y1), 4, (0, 255, 0), -1)
        cv2.circle(out, (line.x2, line.y2), 4, (0, 255, 0), -1)
    return out


# 鉛直線紅、水平線橘，跟 draw_lines 的純紅區隔開
EDGE_COLORS = {'v': (0, 0, 255), 'h': (0, 140, 255)}


def draw_edges(img, edges, thickness=2, label=True):
    """把記錄到的長線畫出來，旁邊標上長度佔畫面寬/高的幾成。"""
    h, w = img.shape[:2]
    out = img.copy()
    for e in edges:
        color = EDGE_COLORS.get(e.axis, (0, 255, 0))
        cv2.line(out, (int(e.x1), int(e.y1)), (int(e.x2), int(e.y2)), color, thickness)
        if label:
            pct = e.length / (h if e.axis == 'v' else w)
            cv2.putText(out, f"{pct:.0%}", (int(e.cx) + 5, int(e.cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def draw_rect(img, quad, color=(0, 255, 0), thickness=2):
    """把四個角畫成一個框。"""
    out = img.copy()
    pts = np.asarray(quad, np.int32).reshape(-1, 1, 2)   # polylines 只吃整數座標
    cv2.polylines(out, [pts], True, color, thickness, cv2.LINE_AA)
    for p in quad:
        cv2.circle(out, (int(p[0]), int(p[1])), 4, color, -1)
    return out


if __name__ == '__main__':
    SRC = r'E:\JProject\JOpenCV\JTest\src\3.jpg'
    OUT_DIR = r'E:\JProject\JOpenCV\JTest\out'
    PCT_V, PCT_H = 0.30, 0.20   # 鉛直線要過高度的 30%、水平線要過寬度的 20%
    os.makedirs(OUT_DIR, exist_ok=True)

    img = cv2.imread(SRC)
    h, w = img.shape[:2]
    edges = record_long_lines(img, pct_v=PCT_V, pct_h=PCT_H)

    print(f"影像 {w}x{h}，門檻 鉛直線 {PCT_V:.0%}×高={PCT_V * h:.0f}px、"
          f"水平線 {PCT_H:.0%}×寬={PCT_H * w:.0f}px")
    print(f"記錄到 {len(edges)} 條夠長的線（鉛直 "
          f"{sum(e.axis == 'v' for e in edges)}、水平 {sum(e.axis == 'h' for e in edges)}）")
    for e in edges:
        span = h if e.axis == 'v' else w
        print(f"   [{e.side:>6}] {e.axis} ({e.x1},{e.y1})-({e.x2},{e.y2})  "
              f"長 {e.length:.0f}px = {e.length / span:.0%}  傾角 {e.angle:+.1f}°  "
              f"連上 {e.fill:.2f}  偏移 {e.dev:.1f}px")

    panel = draw_edges(img, edges)
    cv2.imwrite(os.path.join(OUT_DIR, 'JLongLines.jpg'), panel)

    quad = detect_rect(img, pct_v=PCT_V, pct_h=PCT_H)
    if quad is None:
        print("拼不起矩形")
    else:
        for name, p in zip(('左上', '右上', '右下', '左下'), quad):
            print(f"   {name} ({p[0]:.1f}, {p[1]:.1f})")
        cv2.imwrite(os.path.join(OUT_DIR, 'JRect.jpg'), draw_rect(panel, quad))
    cv2.imshow('long lines / rect', panel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()