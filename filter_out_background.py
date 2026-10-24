"""用「邊界線 + 顏色統計」找文件：提出四角 → 驗證四角 → 收準 → 填遮罩。

這個模組實作的是一個明確的想法，並且把這個想法量到有結論為止：

> 文件跟背景的顏色不同。**統計畫面裡的顏色可以決定主要的顏色和次要的顏色**，
> 把這兩個顏色上色出來，**如果某一塊集中成方框、正方形或矩形，
> 那它就是發票或文件。**

原本的實作是「先上色 → 看有沒有集中成矩形」，實測涵蓋率 0/5，
而且參數一放寬就把背景回傳成文件（數字見 filter_out_background.md 第三節）。
失敗的原因是**上色會滲出**：只要紙跟背景的顏色有一點重疊，
「把這個顏色上色」就會順手把背景畫進去，量到的只剩一個大致長寬。

現在把同一個想法**換到對的位置**：

1. **提出四角** —— 兩種來源都試：
   邊界線（Canny → 閉運算 → 輪廓 → 四邊形）跟分色（Otsu → 連通塊 → 最小外接矩形）。
   輪廓本來就是在解決「分出來的東西邊界不乾淨」這件事。
2. **驗證四角** —— 用顏色統計檢查這個框是不是**真的躺在邊界上**：
   框裡的主色跟「緊貼著框外一圈」的主色必須差得夠多，
   而且框外那一圈不能大多數都跟框裡同色。
   「集中成矩形」改成「框裡框外是兩種不同的顏色」，
   這才擋得住「框到文件上半截就當成整張」那種假陽性。
3. **收準四角** —— 沿著每一條邊往梯度最大的地方拉，把角點收到真正的邊界上。
4. **填出遮罩** —— 框內白、框外黑。

**實測狀況寫在 filter_out_background.md。**
它是 `methods.py` 裡的第三個拉平方法（`color`）。

**要給它原始角度的圖，不要給擺正（旋轉）後的圖**：
旋轉會把畫布撐大、四角補白，那片白會跟白紙連成一塊，
分色時紙跟背景就分不開了。`methods.NEEDS_ALIGNED` 就是在管這件事。
"""

import os
import sys

import cv2
import numpy as np


# ---------------------------------------------------------------- 顏色統計

def dominant_colors(img, n=2, step=16):
    """統計畫面裡佔比最大的 n 種顏色。

    為什麼要先量子化（除到 `step` 的倍數）：JPEG 壓縮會讓同一種顏色
    散成好几個相鄰的數值，直接取「最常見的顏色」會拿到一個只出現幾百次的
    雜訊色。先把每個通道對齊到同一個格子再數，數到的才是真的顏族群。

    回傳 `[(顏色, 佔比 0~1), ...]`，由大到小。灰階時顏色是單一數值。
    """
    if step < 1:
        step = 1
    q = (img // step) * step
    flat = q.reshape(-1, q.shape[-1]) if q.ndim == 3 else q.reshape(-1, 1)
    cols, counts = np.unique(flat, axis=0, return_counts=True)
    total = float(counts.sum())
    order = np.argsort(-counts)[:n]
    return [(tuple(int(v) for v in cols[i].reshape(-1)), float(counts[i]) / total)
            for i in order]


def color_mask(img, color, tol=28, gray=False):
    """把「離 `color` 不太遠」的像素標成 255，其餘 0。

    `img` 是灰階時 `color` 是單一亮度值，用絕對差；
    否則用 BGR 三通道的歐氏距離。兩個都比較寬，
    因為要的是「同一個顏色族群」而不是「同一個數值」。
    """
    if img.ndim == 2:
        return (np.abs(img.astype(np.int16) - int(color[0])) < tol).astype(np.uint8) * 255
    d = np.linalg.norm(img.astype(np.float32) - np.asarray(color, np.float32), axis=2)
    return (d < tol).astype(np.uint8) * 255


def _dominant_gray(vals, step=16):
    """一串灰階值裡的主色（量化後）與它的佔比。

    這裡不用 `dominant_colors`：那個用 `np.unique` 排序，幾十萬個像素很慢，
    而且這個模組只處理灰階，用 `bincount` 直接數就夠了。
    """
    if vals.size == 0:
        return 0, 0.0
    q = (vals.astype(np.int32) // step) * step
    cnt = np.bincount(q, minlength=256)
    i = int(np.argmax(cnt))
    return int(i), float(cnt[i]) / float(vals.size)


def _largest_blob(mask):
    """回傳最大連通塊的 (遮罩, 面積)。沒有就 (None, 0)。"""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        return None, 0
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == k).astype(np.uint8), int(stats[k, cv2.CC_STAT_AREA])


# ---------------------------------------------------------------- 反光

def remove_glare(gray, paper, over=25, radius=21, dilate=5):
    """把比紙還亮很多的像素用 inpaint 補掉。

    **門檻要從紙的亮度算出來，不要寫死。**
    寫死 240 的問題是：白紙本來就接近 240，那樣會把紙本身當成反光塗掉——
    2.png 最常見的顏色是 BGR(240,240,240) 佔 86.7%，等於整張圖被塗掉。
    反光一定是「比紙亮」，所以門檻 = 紙的亮度 + over。

    回傳 `(處理後的圖, 被塗掉的像素比例)`。
    """
    mask = (gray.astype(np.int16) > int(paper) + over).astype(np.uint8) * 255
    frac = float((mask > 0).mean())
    if frac <= 0.0:
        return gray, 0.0
    if dilate > 1:
        mask = cv2.dilate(mask, np.ones((dilate, dilate), np.uint8), iterations=1)
    return cv2.inpaint(gray, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA), frac


# ---------------------------------------------------------------- 矩形檢查

def rect_score(blob, shape, min_area_ratio=0.10, min_fill=0.60, tol_px=0.02):
    """判斷一塊區域有多「集中成矩形」。

    矩形程度只看一個數字：**面積 ÷ 最小外接矩形面積**。

    「內角偏離 90 度」那個條件不能用：最小外接矩形**定義上**就是四個直角，
    實測每一張算出來都是 [90, 90, 90, 90]，那個條件等於沒有條件，
    只會把所有輸入都擋掉。這裡改用面積比，量得到差別。

    另外擋掉「被畫面邊緣切到」的區塊：它確實是個乾淨的矩形，
    但那只是背景的一條而不是文件；而且沿著被切的那個軸，
    我們根本不知道文件的角在哪裡——回傳它的角是錯的。
    """
    ys, xs = np.where(blob > 0)
    area = len(xs)
    h, w = shape[:2]
    if area < min_area_ratio * h * w:
        return {'ok': False, 'why': '面積太小', 'area': area,
                'fill': 0.0, 'rect_fill': 0.0, 'quad': None}

    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    fill = area / float(bw * bh)

    pts = np.column_stack([xs, ys]).astype(np.float32)
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(pts)
    # minAreaRect 量的是「最外側像素中心之間」的距離：N 個像素的跨度是 N−1，
    # 但像素本身覆蓋的面積是 N。差這 1 會讓完美的矩形算出超過 100%
    # （實測 4.jpg 一塊 42×419 的實心長條算出 102.7%），所以要補回來。
    rect_fill = min(1.0, area / max((rw + 1.0) * (rh + 1.0), 1.0))
    quad = cv2.boxPoints(((cx, cy), (rw, rh), ang))

    t = tol_px * max(w, h)
    clipped = ((x0 <= t and x1 >= w - 1 - t) or (y0 <= t and y1 >= h - 1 - t))

    why = []
    if clipped:
        side = '橫向' if (x0 <= t and x1 >= w - 1 - t) else '直向'
        why.append('%s頂到畫面兩邊，是被切掉的背景不是文件' % side)
    if rect_fill < min_fill:
        why.append('最小外接矩形只填滿 %.0f%%' % (rect_fill * 100))
    return {'ok': not why, 'why': '、'.join(why) or 'ok',
            'area': area, 'fill': fill, 'rect_fill': rect_fill,
            'clipped': clipped, 'quad': quad}


# ---------------------------------------------------------------- 四角工具

def _order_quad(pts):
    """把四個點排成左上、右上、右下、左下（順時針一圈）。

    **不要用「x+y 最小 = 左上」那套。** 文件歪 45 度時（5.jpg 正好是），
    四個角在 x+y 跟 x−y 上會打成平手，會排出錯的順序，
    `methods._warp` 拿到就會拉出ㄧ個破掉的矩形。

    改成：先繞中心點用角度排成順時針一圈（螢幕座標 y 往下，角度遞增 = 順時針），
    再從「最上面那個角」開始（同一個高度時取最左邊的）。
    """
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]
    k = int(np.lexsort((pts[:, 0], pts[:, 1]))[0])     # 先比 y，再比 x
    return np.roll(pts, -k, axis=0)


def _corner_dev(quad):
    """四個內角偏離 90 度多少度，取最離譜的那一個。

    這裡量的是**給定的四邊形**，不是最小外接矩形。
    最小外接矩形定義上就是四個直角，拿它來量等於沒有量
    （實測每一張算出來都是 0.0），那個門檻等於沒有門檻。
    """
    worst = 0.0
    for i in range(4):
        p, c, q = quad[i - 1], quad[i], quad[(i + 1) % 4]
        v1, v2 = p - c, q - c
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 999.0
        cosang = float(np.dot(v1, v2) / (n1 * n2))
        dev = abs(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))) - 90.0)
        worst = max(worst, dev)
    return worst


def _covers_frame(quad, shape, tol=0.98):
    """這個框是不是幾乎等於整張畫面。

    不要用「四邊有沒有頂到畫面邊緣」那個判準：**歪的文件本來就會超出畫面**，
    5.jpg 那張發票的左上角就在畫面外面，用那個判準會把真正找對的框殺掉。
    改成量「這個框蓋住畫面的多少」——蓋掉 98% 以上才是「整張畫面就是一塊」。
    """
    h, w = shape[:2]
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [quad.astype(np.int32)], 255)
    return float((m > 0).mean()) >= tol


def _quad_area(quad):
    """四邊形的面積（用鞋帶公式，不是外接框——傾斜的時候差很多）。"""
    x, y = quad[:, 0], quad[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


# ---------------------------------------------------------------- 提出四角

def _edge_candidates(gray, blur=5, canny_lo=36, canny_hi=36,
                     close_ratios=(0.004, 0.010, 0.020), min_area_ratio=0.05,
                     max_cands=10):
    """從邊界線提出四角。

    Canny 出來的邊界線常常是斷的（反光、皺折、紋理），要用閉運算接回去，
    **而且要試好幾種閉運算的強度**：太弱接不起來，太強會把紙跟旁邊的
    水泥地縫隙黏在一起，整張畫面變成一條大輪廓。
    實測 5.jpg：閉 0.004 時發票的輪廓是 50.2%，閉 0.020 就變成 92.5%（整張）。

    `RETR_LIST` 而不是 `RETR_EXTERNAL`：RETR_EXTERNAL 只給最外層，
    一張紙放在桌子中間的話，它的輪廓會被背景的邊緣整個包住而看不到。
    """
    h, w = gray.shape[:2]
    sm = cv2.GaussianBlur(gray, (blur, blur), 0) if blur and blur > 1 else gray
    base = cv2.Canny(sm, canny_lo, canny_hi)

    out = []
    for cr in close_ratios:
        k = max(3, int(round(min(h, w) * cr)))
        edges = cv2.morphologyEx(base, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:max_cands]

        for c in cnts:
            area = cv2.contourArea(c)
            if area < min_area_ratio * h * w:
                continue
            peri = cv2.arcLength(c, True)
            if peri <= 0:
                continue
            # 先試 approxPolyDP：它尊重輪廓真正的形狀，傾斜的紙邊也拿得準。
            # 但邊界線圖上的輪廓是一條細帶子，逼近之後常常是 8 個點
            # （外圈 4 個 + 內圈 4 個），所以要多試幾個 epsilon。
            # 都失敗就退回「凸包 → 最小外接矩形」，那個一定是乾淨的矩形。
            for eps in (0.01, 0.02, 0.03, 0.04, 0.05):
                ap = cv2.approxPolyDP(c, eps * peri, True)
                if len(ap) == 4:
                    pts = ap.reshape(4, 2)
                    if cv2.isContourConvex(pts.astype(np.float32)):
                        out.append((_order_quad(pts), area))
                        break
            hull = cv2.convexHull(c)
            if len(hull) >= 3:
                out.append((_order_quad(cv2.boxPoints(cv2.minAreaRect(hull))), area))
    return out


def _threshold_candidates(gray, close_ratio=0.02, min_area_ratio=0.05,
                          percentiles=(60, 66, 72, 78, 84, 88, 92, 95, 97)):
    """在亮度直方圖上掃一輪門檻，每個門檻都當成「文件是比這個亮的東西」試一次。

    這是「統計主色 → 上色 → 看是否集中成矩形」那個想法**做對之後**的樣子。
    原版用一個寫死的容差（tol=28）圈一個顏色，涵蓋率 0/5，
    放寬參數就會把背景回傳成文件——因為**容差是所有圖共用的**，
    而每張圖的紙跟背景差多少根本不一樣。

    改成**每一張圖自己找門檻**：門檻候選取自亮度直方圖的分位數，
    兩個極性都掃（白紙壓深色桌面、白板上的深色文件）。
    每個門檻各做一次二值化 → 閉運算（把紙上印的字挖掉的洞補起來）
    → 取最大連通塊 → 最小外接矩形。

    順帶量到一件事：門檻要略高於 Otsu。Otsu 的門檻會把受光的水泥地
    也算成「亮的一邊」，框就會往外脹一圈（實測 5.jpg：Otsu 門檻框出 66%，
    往上調一點變成 54%，那才是發票真正的形狀）。
    """
    h, w = gray.shape[:2]
    k = max(3, int(round(min(h, w) * close_ratio)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    thrs = sorted(set([otsu, otsu + 6, max(0, otsu - 6)] +
                      [int(v) for v in np.percentile(gray, percentiles)]))
    out = []
    for thr in thrs:
        for invert in (False, True):
            bw = ((gray <= thr) if invert else (gray >= thr)).astype(np.uint8) * 255
            bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
            n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
            if n < 2:
                continue
            kk = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            blob_area = float(stats[kk, cv2.CC_STAT_AREA])
            if blob_area < min_area_ratio * h * w:
                continue
            cnts, _ = cv2.findContours((lab == kk).astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            c = max(cnts, key=cv2.contourArea)
            out.append((_order_quad(cv2.boxPoints(cv2.minAreaRect(c))), blob_area,
                        thr, invert))
    return out


# ---------------------------------------------------------------- 驗證四角

def _color_profile(gray, quad, step=16, ring_ratio=0.04):
    """檢查這個框是不是**真的躺在邊界上**。回傳 None 代表量不出來。

    這是這個模組真正的判斷。量三件事：

    `in_color`   框裡的主色（量化後的灰階值）
    `dist`       框裡主色跟框外主色差多少灰階
    `same_ring`  緊貼著框外一圈的像素，有多少跟框裡同色

    為什麼要看「框外一圈」而不是整張畫面：整張畫面的框外包含了所有不相干的
    東西，它的統計沒有意義。真正能判斷的是**緊貼著框的那一圈**——
    如果框外那一圈大多數跟框裡同色，代表這個框切在一塊均勻的區域中間，
    那條線是我們自己畫出來的，不是文件真正的邊界。

    這就是「框到文件上半截就當成整張」那種假陽性擋不住的原因：
    框到上半截時，框外那一圈還是同一張紙，`same_ring` 會接近 1。
    """
    h, w = gray.shape[:2]
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [quad.astype(np.int32)], 255)
    inside = gray[m > 0]
    if inside.size < 64:
        return None

    k = max(3, int(round(min(h, w) * ring_ratio)))
    ring_m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    ring = gray[(ring_m > 0) & (m == 0)]
    if ring.size < 64:
        return None

    ci, fi = _dominant_gray(inside, step)
    co, fo = _dominant_gray(ring, step)
    tol = max(4.0, step * 1.5)
    same_ring = float((np.abs(ring.astype(np.int16) - ci) < tol).mean())
    conc_in = float((np.abs(inside.astype(np.int16) - ci) < tol).mean())
    return {'in_color': ci, 'out_color': co, 'dist': abs(ci - co),
            'same_ring': same_ring, 'conc_in': conc_in,
            'in_frac': fi, 'out_frac': fo}


# ---------------------------------------------------------------- 收準四角

def _line_intersect(p1, d1, p2, d2):
    """兩條「點 + 方向」的直線的交點。平行就回傳 None。"""
    A = np.array([d1, -d2]).T
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    t, _ = np.linalg.solve(A, np.asarray(p2, np.float64) - np.asarray(p1, np.float64))
    return np.asarray(p1, np.float64) + t * np.asarray(d1, np.float64)


def _snap_quad(gray, quad, search_ratio=0.08, samples=16, margin=0.15):
    """把四角往真正的邊界拉。

    分色跟輪廓給的框都是「差不多」的位置：門檻邊緣是模糊的，
    遮罩會往外滲十幾個像素（實測 4.jpg 上緣多吃了 12 像素）。
    這裡沿著每一條邊的法線方向找梯度最大的地方，重新把直線接回去，
    交點就是收準後的角。

    取樣只取每條邊中間那段（margin），避免在角上被兩個方向的邊互相干擾。
    """
    h, w = gray.shape[:2]
    search = max(2, int(round(min(h, w) * search_ratio)))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    lines = []
    for i in range(4):
        p0, p1 = quad[i], quad[(i + 1) % 4]
        d = p1 - p0
        L = float(np.linalg.norm(d))
        if L < 4.0:
            return quad
        t = d / L
        nrm = np.array([-t[1], t[0]])          # 法線
        pts = []
        for s in np.linspace(margin, 1.0 - margin, samples):
            base = p0 + d * s
            best, bestv = None, -1.0
            for off in range(-search, search + 1):
                q = base + nrm * off
                xi, yi = int(round(q[0])), int(round(q[1]))
                if not (0 <= xi < w and 0 <= yi < h):
                    continue
                v = float(mag[yi, xi])
                if v > bestv:
                    bestv, best = v, q
            if best is not None:
                pts.append(best)
        if len(pts) < 4:
            return quad
        vx, vy, x0, y0 = cv2.fitLine(np.asarray(pts, np.float32),
                                     cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0], np.float64), np.array([vx, vy], np.float64)))

    out = []
    for i in range(4):
        p, d = lines[i - 1]
        q, e = lines[i]
        ip = _line_intersect(p, d, q, e)
        if ip is None:
            return quad
        out.append(ip)
    snapped = _order_quad(np.array(out, np.float64))

    # 收準不能跑太遠，否則就是被別的邊緣拐跑了。
    if np.abs(snapped - quad).max() > search * 1.5:
        return quad
    return snapped


def _edge_strength(gray, quad, samples=24, margin=0.10):
    """沿著四條邊取樣的平均梯度大小。

    用來判斷「收準之後是不是落在更銳利的邊上」。
    光看顏色統計不夠：框挪動十幾個像素，顏色統計幾乎不變，
    但「框正好貼在紙邊上」跟「框停在旁邊的空地上」差很多。
    """
    h, w = gray.shape[:2]
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    vals = []
    for i in range(4):
        p0, p1 = quad[i], quad[(i + 1) % 4]
        d = p1 - p0
        if float(np.linalg.norm(d)) < 4.0:
            continue
        for s in np.linspace(margin, 1.0 - margin, samples):
            q = p0 + d * s
            xi, yi = int(round(q[0])), int(round(q[1]))
            if 0 <= xi < w and 0 <= yi < h:
                vals.append(float(mag[yi, xi]))
    return float(np.mean(vals)) if vals else 0.0


def _refine(gray, quad, prof, snap_ratios, min_color_dist, max_same_ring, step=16):
    """把框收到真正的邊界上，並且只接受「落在更銳利的邊上」的那個結果。

    收準要同時滿足兩個條件才採用：
      1. 顏色判斷沒有變差（框裡框外的對比沒有變弱、框外那圈沒有變同色）
      2. 沿著四條邊的平均梯度變大了 —— 也就是說框確實挪到了更銳利的邊上

    只看條件 1 不夠：框挪動十幾個像素，顏色統計幾乎不變，
    但「框正好貼在紙邊上」跟「框停在旁邊的空地上」差很多。
    """
    base_strength = _edge_strength(gray, quad)
    best = (base_strength, quad, prof)
    for r in snap_ratios:
        snapped = _snap_quad(gray, quad, search_ratio=r)
        if snapped is quad or np.abs(snapped - quad).max() < 1e-6:
            continue
        p = _color_profile(gray, snapped, step=step)
        if p is None:
            continue
        if p['dist'] < 0.9 * prof['dist'] or p['same_ring'] > max_same_ring:
            continue
        strength = _edge_strength(gray, snapped)
        if strength > best[0]:
            best = (strength, snapped, p)
    return best[1], best[2]


# ---------------------------------------------------------------- 主流程

def detect_document_by_color(img, work=800, blur=5, canny_lo=36, canny_hi=36,
                             close_ratio=0.02, min_area_ratio=0.05,
                             max_area_ratio=0.92, max_corner_dev=35.0,
                             min_extent=0.25, step=16, min_color_dist=40.0,
                             max_same_ring=0.40, snap=True,
                             snap_ratios=(0.04, 0.08, 0.12),
                             remove_glare_=False, glare_over=25, glare_radius=21):
    """找文件四角，並填出「內白外黑」的遮罩。回傳 `(四角 或 None, info)`。

    四角順序是左上、右上、右下、左下，可以直接餵給 `methods._warp`。
    找不到就回傳 `None` —— 跟其他方法一樣，寧可承認做不到。

    `info` 裡有：
      `mask`  內白外黑的遮罩（跟原圖同尺寸）
      `try`   每個候選被刷掉的理由
      `note`  一句話結論
    """
    h0, w0 = img.shape[:2]
    sc = work / float(max(h0, w0))
    # 圖本來就比 `work` 小的時候**不要縮**，而且座標也不要除倍率——
    # 沒有縮過就沒有倍率可除，除了座標會整個變小（實測 4.jpg 整張縮成一半）。
    if sc < 1.0:
        small = cv2.resize(img, (max(1, int(round(w0 * sc))), max(1, int(round(h0 * sc)))),
                            interpolation=cv2.INTER_AREA)
        back = 1.0 / sc
    else:
        small = img
        back = 1.0

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small.copy()
    hh, ww = gray.shape[:2]

    glare = 0.0
    if remove_glare_:
        paper, _ = _dominant_gray(gray.reshape(-1), step)
        gray, glare = remove_glare(gray, paper, over=glare_over, radius=glare_radius)

    # 候選：門檻掃描（主色那一路）＋邊界線（輪廓那一路）
    cands = []
    for quad, blob_area, thr, invert in _threshold_candidates(
            gray, close_ratio=close_ratio, min_area_ratio=min_area_ratio):
        cands.append((quad, blob_area, '門檻%d%s' % (thr, '暗' if invert else '亮')))
    for quad, area in _edge_candidates(gray, blur=blur, canny_lo=canny_lo,
                                       canny_hi=canny_hi,
                                       min_area_ratio=min_area_ratio):
        cands.append((quad, area, '邊界線'))

    tried = []
    best = None
    for quad, blob_area, src in cands:
        qa = _quad_area(quad)
        ratio = qa / float(hh * ww)
        if ratio < min_area_ratio or ratio > max_area_ratio:
            continue
        if _corner_dev(quad) > max_corner_dev:
            continue
        if _covers_frame(quad, (hh, ww)):
            continue
        qw = float(quad[:, 0].max() - quad[:, 0].min())
        qh = float(quad[:, 1].max() - quad[:, 1].min())
        if qw < min_extent * ww or qh < min_extent * hh:
            continue

        prof = _color_profile(gray, quad, step=step)
        if prof is None:
            continue
        if prof['dist'] < min_color_dist:
            tried.append('%s 框裡主色 %d、框外主色 %d 只差 %d，不足以證明這是邊界'
                         % (src, prof['in_color'], prof['out_color'], prof['dist']))
            continue
        if prof['same_ring'] > max_same_ring:
            tried.append('%s 框外一圈有 %.0f%% 跟框裡同色，這條線是畫出來的'
                         % (src, prof['same_ring'] * 100))
            continue

        # 「框有多貼合那塊區域」：區塊面積 ÷ 框面積。
        # 框如果比區塊大很多，代表框邊上有一截不是東西（實測 5.jpg 的陰影）。
        fill = (blob_area / float(hh * ww)) / max(ratio, 1e-6)
        score = prof['dist'] * (1.0 - prof['same_ring']) * prof['conc_in'] * fill
        tried.append('%-8s 框 %dx%d 面積 %.0f%% 貼合 %5.1f%%  主色 %d / 框外 %d  '
                     '差 %d  框外同色 %.0f%%  內集中 %.0f%%  分數 %.0f'
                     % (src, qw, qh, ratio * 100, fill * 100, prof['in_color'],
                        prof['out_color'], prof['dist'], prof['same_ring'] * 100,
                        prof['conc_in'] * 100, score))
        if best is None or score > best[0]:
            best = (score, quad, prof, src, blob_area)

    if best is None:
        return None, {'found': False, 'try': tried, 'glare': glare,
                      'note': '找不到通過顏色檢查的四角'}

    score, quad, prof, src, blob_area = best

    if snap:
        quad, prof = _refine(gray, quad, prof, snap_ratios,
                              min_color_dist, max_same_ring, step)

    mask_small = np.zeros((hh, ww), np.uint8)
    cv2.fillPoly(mask_small, [quad.astype(np.int32)], 255)

    # 遮罩要跟原圖同尺寸才能直接套在原圖上。
    mask = cv2.resize(mask_small, (w0, h0), interpolation=cv2.INTER_NEAREST) \
        if (h0, w0) != (hh, ww) else mask_small

    quad_full = quad * back
    pts = ' '.join('(%d,%d)' % (p[0], p[1]) for p in quad_full)
    note = ('%s %s  框內主色 %d、框外主色 %d  面積 %.0f%%'
            % (src, pts, prof['in_color'], prof['out_color'],
               _quad_area(quad) / float(hh * ww) * 100))
    return quad_full, {'found': True, 'try': tried, 'glare': glare,
                       'score': score, 'profile': prof, 'source': src,
                       'mask': mask, 'note': note}


def fill_document_by_color(img, **kwargs):
    """`detect_document_by_color` 的別名，回傳 `(四角, info)`。

    `info['mask']` 是「內白外黑」的遮罩，跟原圖同尺寸，
    可以直接餵給 `apply_mask` 拿到填好的圖。
    """
    return detect_document_by_color(img, **kwargs)


# ---------------------------------------------------------------- 舊版（留著比對用）

def detect_document_by_stats(img, n_colors=2, step=16, tol=28, blur=5,
                             work=600, close_ratio=0.02, min_area_ratio=0.10,
                             min_fill=0.60, tol_px=0.02,
                             remove_glare_=False, glare_over=25, glare_radius=21):
    """**舊版**：直接用主色／次色上色，看有沒有集中成矩形。回傳 `(四角 或 None, info)`。

    這是「先上色 → 檢查矩形」那個原始想法的實作，**涵蓋率 0/5**，
    參數一放寬就會把背景回傳成文件（數字見 filter_out_background.md 第三節）。
    留在這裡是為了讓那些數字還跑得回來；
    專案真正在用的是 `detect_document_by_color`。

    四角順序是左上、右上、右下、左下。找不到就回傳 `None`。
    """
    h0, w0 = img.shape[:2]
    sc = work / float(max(h0, w0))
    if sc < 1.0:
        small = cv2.resize(img, (max(1, int(round(w0 * sc))), max(1, int(round(h0 * sc)))),
                           interpolation=cv2.INTER_AREA)
        back = 1.0 / sc
    else:
        small = img
        back = 1.0

    view = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small.copy()
    view = cv2.threshold(view, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    if blur and blur > 1:
        view1 = cv2.medianBlur(view, 7)
        view = cv2.GaussianBlur(view1, (blur, blur), 0)

    colors = dominant_colors(view, n=n_colors, step=step)

    glare = 0.0
    if remove_glare_ and colors:
        base_c = colors[0][0][0] if isinstance(colors[0][0], (tuple, list, np.ndarray)) else colors[0][0]
        view, glare = remove_glare(view, base_c, over=glare_over, radius=glare_radius)
        colors = dominant_colors(view, n=n_colors, step=step)

    hh, ww = view.shape[:2]
    k = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(3, int(min(hh, ww) * close_ratio)),) * 2)

    tried = []
    for rank, (color, frac) in enumerate(colors):
        m = color_mask(view, color, tol=tol, gray=True)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

        blob, _ = _largest_blob(m)
        if blob is None:
            c_val = color[0] if isinstance(color, (tuple, list, np.ndarray)) else color
            tried.append('第 %d 色 灰度%3d（%4.1f%%）：沒有區塊' % (rank + 1, c_val, frac * 100))
            continue

        r = rect_score(blob, view.shape, min_area_ratio, min_fill, tol_px)
        r.update(rank=rank + 1, color=color, frac=frac, mask=blob)

        c_val = color[0] if isinstance(color, (tuple, list, np.ndarray)) else color
        tried.append('第 %d 色 灰度%3d（%4.1f%%）：面積%5.1f%% 最小外接矩形填滿%5.1f%%'
                     ' 軸對齊填滿%5.1f%% %s'
                     % (rank + 1, c_val, frac * 100,
                        r['area'] / float(hh * ww) * 100,
                        r['rect_fill'] * 100, r['fill'] * 100,
                        r['why'] if not r['ok'] else '→ 是矩形'))
        if r['ok']:
            quad = _order_quad(r['quad'])
            return quad * back, {
                'found': True, 'try': tried, 'score': r, 'glare': glare,
                'note': '第 %d 色 灰度%3d（%4.1f%%）集中成矩形，最小外接矩形填滿%5.1f%%'
                        % (rank + 1, c_val, frac * 100, r['rect_fill'] * 100)}

    return None, {'found': False, 'try': tried, 'glare': glare,
                  'note': '%d 種主色都不集中成矩形' % len(colors)}


# ---------------------------------------------------------------- 輸出

def apply_mask(img, mask, bg=(255, 255, 255)):
    """把遮罩以外的像素換成 `bg`，留下文件本身。

    這就是「填出來」的那張圖：文件留在原位，框外塗成背景色。
    """
    out = img.copy()
    if out.ndim == 3 and mask.ndim == 2:
        bgv = (bg,) if isinstance(bg, int) else tuple(bg)
    else:
        bgv = bg
    out[mask == 0] = bgv
    return out


def draw(img, quad=None, mask=None):
    """把找到的框畫在原圖上：遮罩輪廓黃色、找出的框紅色。"""
    out = img.copy()
    if mask is not None:
        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (0, 255, 255), 2)
    if quad is not None:
        flat = np.asarray(quad).reshape(-1)
        # 防禦：座標數量不對就不畫，寧可少畫也不要拋錯
        if len(flat) >= 8 and len(flat) % 2 == 0:
            pts = flat[:len(flat) // 2 * 2].reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(out, [pts], True, (0, 0, 255), 3)
    return out


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')   # Windows 主控台預設不是 utf-8
    src = sys.argv[1] if len(sys.argv) > 1 else 'src'
    out_dir = sys.argv[2] if len(sys.argv) > 2 else 'out'
    glare = '--glare' in sys.argv           # 要比較「有沒有移除反光」就打這個
    stats = '--stats' in sys.argv          # 跑舊版（主色上色）比對

    try:
        from detect_angle_rect_only import detect_quad
    except Exception:
        detect_quad = None

    fn = detect_document_by_stats if stats else detect_document_by_color
    print('%s%s\n' % ('舊版：主色上色 → 矩形檢查' if stats else '邊界線/分色 → 四角 → 顏色驗證',
                      '（含移除反光）' if glare else ''))
    for name in sorted(os.listdir(src)):
        if os.path.splitext(name)[1].lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
            continue
        im = cv2.imread(os.path.join(src, name))
        if im is None:
            continue
        quad, info = fn(im, remove_glare_=glare)
        stem = os.path.splitext(name)[0]
        print('%s  %dx%d  移除反光的像素 %.1f%%'
              % (name, im.shape[1], im.shape[0], info['glare'] * 100))
        for t in info['try']:
            print('    ' + t)
        print('    → ' + info['note'])

        if detect_quad is not None:
            ref = detect_quad(im)
            same = (quad is not None and ref is not None
                    and np.abs(np.asarray(quad) - np.asarray(ref)).max() < 20)
            print('    對照 detect_quad: %s'
                  % ('相同' if ref is not None and same else
                     ('它找到' if ref is not None else '它也找不到')))
        print()

        os.makedirs(out_dir, exist_ok=True)
        mask = info.get('mask')
        if mask is not None:
            cv2.imwrite(os.path.join(out_dir, stem + '_mask.jpg'), mask)
            cv2.imwrite(os.path.join(out_dir, stem + '_fill.jpg'),
                        apply_mask(im, mask, bg=(255, 255, 255)))
        cv2.imwrite(os.path.join(out_dir, stem + '_box.jpg'), draw(im, quad, mask))
