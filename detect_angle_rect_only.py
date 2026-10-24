"""只量角度：找出文件在畫面裡歪了幾度。

這是 `detect_lines_and_rect_cv` 的簡化版。那個模組要找出紙張的邊界線、
把斷掉的線接起來、再用交點算出四角；這個模組只看一件事——紙歪了幾度，
不需要線、不需要角點，也不需要 Tesseract。

**回傳的角度是模 90 的。** 這不是偷懶，是幾何上的必然：一個矩形轉 θ 度
跟轉 θ+90 度，在邊緣上看起來一模一樣（左右兩條邊換成上下兩條邊而已）。
光看邊緣沒有辦法分辨「紙是正拍」跟「紙是橫拍」，要分辨得靠認字，
那是 `auto_align_rotate` 的工作。所以這個模組回傳的值域是 −45°～+45°。

做法與實測數據的說明在 detect_angle_rect_only.md。
"""

import os
import sys

import cv2
import numpy as np


def _peak(a, bins, bin_deg, guard_bins):
    """把角度做成直方圖取最高峰。回傳 (內插後的角度, 峰高, 對手峰高)。

    「對手」要排除贏家正後方那一小段：峰頂所在區間的左右鄰居本來就跟它
    是同一個角度家族（同一條紙邊被切成好幾段），不算競爭者。
    真正的競爭者是 90 度外的那一組線——文字行跟紙的左右邊界之類，
    數量最多的那個才是真正排紙的那一族。
    """
    hist, _ = np.histogram(a, bins=bins)
    i = int(np.argmax(hist))
    peak = int(hist[i])
    lo, hi = max(0, i - guard_bins), min(len(hist), i + guard_bins + 1)
    rest = np.concatenate((hist[:lo], hist[hi:]))
    rival = int(rest.max()) if rest.size else 0

    angle = float(bins[i] + bin_deg / 2)
    if 0 < i < len(hist) - 1:
        y0, y1, y2 = hist[i - 1], hist[i], hist[i + 1]
        den = y0 - 2 * y1 + y2
        if den:
            angle += bin_deg * 0.5 * (y0 - y2) / den
    return angle, peak, rival


def detect_angle(img, bin_deg=0.25, min_len_ratio=0.08, max_line_gap_ratio=0.02,
                 threshold_ratio=0.08, lo=50, hi=150,
                 batch=100, max_rounds=10, min_peak=20, lead_ratio=1.5):
    """回傳讓文件轉正所需要的旋轉角度（度，正值＝逆時針），值域 −45～+45。

    量不到任何線的時候回傳 0.0。

    旋轉方向跟 cv2.getRotationMatrix2D 一致：回傳 +30 就是把圖轉 +30 度。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape[:2]
    short = min(h, w)

    # 門檻全部照影像短邊的比例給，不要寫死。
    edges = cv2.Canny(gray, lo, hi, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=max(20, int(short * threshold_ratio)),
                            minLineLength=int(short * min_len_ratio),
                            maxLineGap=int(short * max_line_gap_ratio))
    if lines is None or len(lines) == 0:
        return 0.0

    seg = lines.reshape(-1, 4).astype(np.float64)      # OpenCV 4 回 (N,1,4)、5 回 (N,4)
    dx, dy = seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]
    keep = np.hypot(dx, dy) > 1e-6
    a = np.degrees(np.arctan2(dy[keep], dx[keep]))
    a = (a + 90) % 180 - 90                            # 線轉 180 度還是同一條線

    # 一口氣把全部線灌進直方圖會用到兩千多條，看起來是「資料越多越好」，
    # 其實多出來的那些大部分是文件裡的印刷字和框線——它們的角度跟紙邊無關，
    # 只是讓直方圖變熱，卻不會讓真正的那一族變得更突出。
    # 改成每次只吃 batch 條，確定了就停，最多吃 max_rounds 批。
    #
    # 分批之前先重排一次：HoughLinesP 回傳的順序沒有統計意義，
    # 直接取前 N 條會拿到偏斜的樣本。跨整批平均取樣，讓每一批都攤到
    # 全部的角度分布，這樣第一批 100 條就有代表性。
    #
    # 實測五張圖的角度跟「全部一次灌進去」一模一樣（誤差 0.00 度），
    # 差別在用量：2.png、4.jpg、5.jpg 第一批 100 條就夠確定，直接停。
    n = len(a)
    rounds = min(max_rounds, max(1, -(-n // batch)))
    if rounds > 1:
        order = np.argsort((np.arange(n) * rounds) // n, kind='stable')
        a = a[order]

    bins = np.arange(-90, 90 + bin_deg, bin_deg)
    guard = max(2, int(round(2.5 / bin_deg)))          # ±2.5 度算同一個家族
    angle, peak, rival = _peak(a[:batch], bins, bin_deg, guard)
    for r in range(2, rounds + 1):
        # 還不夠確定就再吃一批。兩個條件同時成立才停：
        #   峰高 ≥ min_peak    —— 這個角度確實有一堆線同意
        #   峰高 ≥ lead×對手   —— 而且明顯贏過 90 度外的那一組線
        if peak >= min_peak and peak >= lead_ratio * rival:
            break
        angle, peak, rival = _peak(a[:min(n, r * batch)], bins, bin_deg, guard)

    return float((angle + 45) % 90 - 45)


def _order_quad(pts):
    """把四個點排成左上、右上、右下、左下。

    座標系的 y 往下，所以「上」是 y 最小的那兩個，「左」是 x 最小的那兩個。
    """
    pts = np.asarray(pts, np.float64)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)],      # 左上：x+y 最小
                     pts[np.argmin(d)],      # 右上：x−y 最小
                     pts[np.argmax(s)],      # 右下：x+y 最大
                     pts[np.argmax(d)]],    # 左下：x−y 最大
                    np.float64)


def _corner_angles_ok(quad, max_dev=35.0):
    """四個內角可不可以接受（離 90° 多少度以內）。

    這是排除「框到處都是，只是形狀怪」的主要關卡。
    """
    n = len(quad)
    for i in range(n):
        p, c, q = quad[i - 1], quad[i], quad[(i + 1) % n]
        v1, v2 = p - c, q - c
        if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
            return False
        cosang = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        if abs(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))) - 90.0) > max_dev:
            return False
    return True


def _spans_frame(quad, w, h, tol=0.02):
    """這個框是不是等於整張畫面（四邊都頂到）。

    為什麼要特別排掉：Otsu 分出來最大的那一塊通常是**背景**不是文件，
    因為背景常常比較大。3.jpg 的背景佔畫面 93%，直接取最大塊會拿到背景，
    而背景是繞著收據一圈的不規則形狀，不是乾淨的矩形。
    四邊都頂到畫面邊緣 = 「整張畫面就是一塊」，那不是偵測到的文件。
    """
    t = tol * max(w, h)
    return (quad[:, 0].min() <= t and quad[:, 0].max() >= w - 1 - t and
            quad[:, 1].min() <= t and quad[:, 1].max() >= h - 1 - t)


def _candidate_quads(contour, approx, approx_eps, peri):
    """依序給出這一塊輪廓的幾種「四角」算法，由準到粗。

    順序有意義：先給 approxPolyDP 的結果（尊重輪廓真正的形狀），
    再給最小外接矩形（保底，一定是乾淨的矩形）。
    """
    out = []
    if len(approx) == 4 and cv2.isContourConvex(approx.astype(np.float32)):
        out.append(approx.reshape(4, 2).astype(np.float64))
    if peri > 0:
        out.append(cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float64))
    return out


def detect_quad(img, min_area_ratio=0.10, approx_eps=0.02, max_corner_dev=35.0,
                min_extent=0.35, min_solidity=0.75, work_side=1200, blur=5,
                close_ratio=0.02):
    """用「文件跟背景的明暗差」找出四角。回傳 (4, 2) 左上右上右下左下；找不到回傳 None。

    跟 `detect_lines_and_rect_cv` 完全是兩條路：

    - 那邊認的是**邊界線**：沿掃描線把邊緣點一點一點長成線，再用交點算角。
    - 這邊認的是**明暗分界**：文件通常是亮的一張紙压在較暗的桌面上，
      先用 Otsu 把畫面分成兩塊，再從塊的形狀讀出四個角。

    為什麼要分兩種極性各試一次：白紙壓深色桌面是前者，
    但白板上的深色文件、或是熱感應收據印在白紙上放到白桌上，
    兩邊都接近亮，這時候要靠「哪一塊比較像一張紙」來選，
    所以兩種都算、各自取面積最大的輪廓，最後挑可信度比較高的。

    之前這個函式是「窮舉框的位置、看哪個框的邊緣最密」。
    那條路走不通，實測 5.jpg 找出來的框落在發票表格的內部：
    表格裡每一條線都是邊緣，密到最高分，而文件的輪廓反而不是邊緣最密的地方。
    **光看邊緣密度找不到文件的輪廓，一定要有明暗對比。**
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h0, w0 = gray.shape[:2]

    # 先縮小再找，最後把座標乘回來。輪廓分析對解析度很敏感，
    # 但原圖那麼大只是浪費時間，1200 像素足以分出紙和桌子。
    scale = work_side / float(max(h0, w0))
    work = cv2.resize(gray, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_AREA) if scale < 1.0 else gray
    work = cv2.GaussianBlur(work, (blur, blur), 0)
    hh, ww = work.shape[:2]

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(3, int(min(hh, ww) * close_ratio)),) * 2)

    best = None
    best_score = -1.0
    for invert in (False, True):
        mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, bw = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY | mode | cv2.THRESH_OTSU)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cands = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < min_area_ratio * hh * ww:
                continue
            peri = cv2.arcLength(c, True)
            if peri <= 0:
                continue
            approx = cv2.approxPolyDP(c, approx_eps * peri, True)

            # 兩種拿四角的方式都試，哪個通過檢查就用哪個。
            #
            # 只用 approxPolyDP 會漏掉 5.jpg：那張發票是摺過、有皺紋的，
            # 外框被 approxPolyDP 簡化成四個點，但四個內角都不是 90 度
            # （折痕把角拉歪了），整個候選就被丟掉——明明紙就好好地在那裡。
            # 最小外接矩形是「包住這塊輪廓的最小矩形」，它一定是乾淨的矩形，
            # 對皺掉、彎折、被陰影咬一口的紙都還成立。
            for pts in _candidate_quads(c, approx, approx_eps, peri):
                quad = _order_quad(pts)
                if not _corner_angles_ok(quad, max_dev=max_corner_dev):
                    continue
                if _spans_frame(quad, ww, hh):
                    continue                 # 整張畫面，不算偵測到文件
                # 兩個方向都要夠大。3.jpg 的收據在 Otsu 之後被切成兩塊，
                # 只拿到下面那塊 200×273 的碎片：寬度佔畫面 50% 看起來合格，
                # 高度只有 37%。文件不會只佔畫面這麼窄一條，
                # 這種「其中一邊太瘦」的一定是碎片，不是文件。
                if ((quad[:, 0].max() - quad[:, 0].min()) < min_extent * ww or
                        (quad[:, 1].max() - quad[:, 1].min()) < min_extent * hh):
                    continue
                # 充實度 = 輪廓面積 ÷ 它自己外接矩形的面積。
                # 一張紙是實心的一塊，這個值接近 1（實測背景 0.84）；
                # 被切碎的碎片裡是空的，只有 0.55（實測 3.jpg 的殘塊）。
                # 這一道擋掉「形狀像矩形但其實是別的東西挖掉一塊」。
                box_area = float((quad[2][0] - quad[0][0]) * (quad[2][1] - quad[0][1]))
                if area < min_solidity * box_area:
                    continue
                cands.append((area, quad))
                break                       # 這一塊已經有可用的四角了

        if not cands:
            continue
        area, quad = max(cands, key=lambda t: t[0])
        # 分數 = 面積占比。兩種極性都跑，看誰佔掉畫面比較多。
        score = area / float(hh * ww)
        if score > best_score:
            best_score = score
            best = quad

    if best is None:
        return None
    return best / scale if scale < 1.0 else best


def angle_matrix(shape, angle):
    """回傳 (旋轉矩陣, 新畫布大小 (寬, 高))。

    旋轉的畫布會跟著放大、內容置中，這個矩陣同時決定了
    「像素搬到哪裡」跟「畫布變多大」。`apply_angle` 用它轉圖，
    `methods._rotate_quad` 用它轉角點——**兩邊必須是同一份**，
    角點才會真的落在對的位置上（實測標記點與 M @ p 完全相符）。

    沒有給角度（0）就原樣不動，畫布維持原大小。
    """
    h, w = shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw = int(np.ceil(h * sin + w * cos))
    nh = int(np.ceil(h * cos + w * sin))
    M[0, 2] += nw / 2.0 - cx
    M[1, 2] += nh / 2.0 - cy
    return M, (nw, nh)


def apply_angle(img, angle):
    """照 detect_angle 回傳的角度把圖轉正。畫布會跟著放大，避免切掉四角。

    放大出來的四角補白。BORDER_REPLICATE（把邊緣像素拉出去填）在角落會拖出
    一條條拉長的邊緣色帶，看起來像髒邊，所以改成 BORDER_CONSTANT + 白。
    """
    if not angle:
        return img
    M, (nw, nh) = angle_matrix(img.shape[:2], angle)
    # borderValue 一定要給三元組。實測：給純量 255，OpenCV 會當成
    # Scalar(255, 0, 0)，在 BGR 上就是藍色，不是白色。
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(255, 255, 255))


def _prep_variants(gray):
    """三種「先弄成灰階」的方式，用來比分色的效果。

    平場校正 = 除掉大範圍的光照。同一張紙從頭到尾亮度不一樣，
    亮度直方圖就會被抹開成一個寬峰而不是兩個峰。
    均勻化 = 把對比拉到最大，看把文字跟背景拉開有沒有用。
    """
    illum = cv2.GaussianBlur(gray, (0, 0), 60)
    flat = np.clip(gray.astype(np.float32) / np.maximum(illum.astype(np.float32), 1) * 128,
                   0, 255).astype(np.uint8)
    return (('原灰階', gray),
            ('平場校正', flat),
            ('均勻化', cv2.equalizeHist(gray)))


def _largest_blob(mask):
    """回傳最大連通塊的 (面積比, 外接框, 矩形填滿度)；沒有區塊就 None。"""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, a = stats[k]
    return a / float(mask.size), (x, y, w, h), a / float(w * h)


def why_quad_fails(img, work=400):
    """診斷 `detect_quad` 為什麼在這張圖上找不到四角。**只是診斷，不改變行為。**

    `detect_quad` 失敗只有兩個原因，這裡把它們量成可比較的數字：

    1. 紙跟背景在亮度上分不開 → Otsu 分出來的最大塊是**整張畫面**
    2. 文件本來就占滿整張畫面 → 同上，但要看的框等於整張圖

    順便回答「先轉灰階 / 先壓平光照 / 先均勻化 會不會比較好」。
    實測答案是不會：三種都量了，理由與數字見 detect_angle_rect_only.md。

    回傳一個字典，欄位是 size / color / spread / hist16 / variants。
    """
    h, w = img.shape[:2]
    s = work / float(max(h, w))
    small = cv2.resize(img, (int(w * s), int(h * s))) if s < 1.0 else img

    # 三通道有沒有差。全 0 就是灰階圖，顏色的方法完全沒有資訊可用。
    if small.ndim == 3:
        ch = small.astype(np.int16)
        spread = int(max(np.abs(ch[:, :, 0] - ch[:, :, 1]).max(),
                         np.abs(ch[:, :, 1] - ch[:, :, 2]).max(),
                         np.abs(ch[:, :, 0] - ch[:, :, 2]).max()))
    else:
        spread = 0

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    hist = cv2.calcHist([gray], [0], None, [16], [0, 256]).ravel()

    variants = []
    for tag, im in _prep_variants(gray):
        _, m = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        blob = _largest_blob((m > 0).astype(np.uint8))
        if blob is None:
            variants.append((tag, None))
            continue
        area, box, fill = blob
        # 外接框吃掉九成以上 = 分不出形狀，這張圖對 Otsu 沒有用
        whole = box[2] * box[3] >= m.size * 0.92
        variants.append((tag, {'area': area, 'bbox': box, 'fill': fill, 'whole': whole}))

    return {'size': (w, h), 'color': spread > 0, 'spread': spread,
            'hist16': ' '.join('%4.0f' % (v / hist.sum() * 100) for v in hist),
            'variants': variants}


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')     # Windows 主控台預設不是 utf-8，中文會變亂碼
    src = sys.argv[1] if len(sys.argv) > 1 else 'src'
    for name in sorted(os.listdir(src)):
        if os.path.splitext(name)[1].lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
            continue
        im = cv2.imread(os.path.join(src, name))
        if im is None:
            continue
        angle = detect_angle(im)
        quads = detect_quad(im)
        print(f'{name:12s} {im.shape[1]}x{im.shape[0]:<6d} 角度 {angle:+6.2f}° 四角 {quads}')
        if quads is not None:
            continue
        # 找不到四角就把原因印出來，不然每次都要重猜
        r = why_quad_fails(im)
        print('             診斷 %s  通道最大差 %d'
              % ('彩色圖' if r['color'] else '灰階圖（沒有顏色資訊）', r['spread']))
        print('             亮度 16 桶 %s' % r['hist16'])
        for tag, v in r['variants']:
            if v is None:
                print('             %-10s 沒有區塊' % tag)
                continue
            mark = '  ← 整張畫面，紙跟背景在亮度上分不開' if v['whole'] else ''
            print('             %-10s 面積%5.1f%%  框 %dx%d  矩形填滿%5.1f%%%s'
                  % (tag, v['area'] * 100, v['bbox'][2], v['bbox'][3],
                     v['fill'] * 100, mark))
        print()