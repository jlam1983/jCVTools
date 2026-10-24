"""幾個方法的共用介面。

專案裡有兩組「做出同一件事」的方法，各有各的強弱。這裡只是把它們
對應到名字，讓上層可以用字串切換，不用知道底下是哪個演算法。

**對齊旋轉**（決定照片歪了幾度、該不該翻轉）

| 名稱 | 認什麼 | 優點 | 缺點 |
|------|--------|------|------|
| `geometry` | 邊緣線的角度（霍夫） | 快、免 Tesseract | 角度只確定到模 90 |
| `text` | 墨水的投影輪廓 + 認字 | 連模 90 的正負號、上下翻轉都處理 | 慢、要 Tesseract |

**拉平整頁**（找出文件的四角、拉平、湊成頁面）

| 名稱 | 認什麼 | 優點 | 缺點 |
|------|--------|------|------|
| `lines` | 沿掃描線長出來的邊界線 | 紙邊斷掉還是找得到 | 對邊緣對比弱的紙無解 |
| `contrast` | 文件跟背景的明暗分界 | 快 4000 倍 | 紙跟背景亮度接近就沒用 |
| `color` | 門檻掃描 + 邊界線，再用顏色統計驗證 | 認的是顏色不是形狀 | 紙跟背景顏色接近就沒用 |

## 介面約定

各個方法都是獨立的函式，可以直接 import 直接呼叫，不經過這裡。
它們只負責算出**幾何量**（歪幾度、要不要翻、四角在哪），不負責轉圖也不負責排版，
那是介面層的事。

- **對齊** `align(img)` → `(擺正後的圖, info)`，`info` 有 `skew`、`turn`、`note`
- **拉平** `straighten(img)` → `(湊好頁面的圖 或 None, info)`，
  找不到四角就回傳 `None`，`info` 有 `quad`、`note`

回傳 `None` 而不是硬掰一個框，是刻意的：一個亂猜的四角會讓輸出看起來
「有處理過」但其實裁錯，比直接承認做不到更麻煩。

`scan` / `scan_auto` 會把兩層的資訊合併，這時兩個 `note` 分別改名成
`align_note` 與 `straighten_note`，避免後者蓋掉前者。

## 縮圖偵測

`scan` / `scan_auto` 不在原圖上偵測，而是：

1. 縮到 `WORK_SIZE` 的框內（等比例，不變形）
2. **在小圖上**量角度、找四角
3. 旋轉、裁切、輸出一律做在**原圖**上，所以解析度不會因為縮圖而損失

角度是無號的，不用換算；四角是座標，要乘回去。
倍率用旋轉後兩張圖的實際尺寸反推，不靠縮放時算出來的比例——
`apply_angle` 的新畫布大小有 ceil 會差一像素，用算出來的比例會差那麼一點。

**不是每個方法都在擺正後的圖上跑。** `NEEDS_ALIGNED` 決定這件事：
靠亮度分文件跟背景的（`contrast`、`color`）要在**原始角度**的圖上跑，
因為旋轉補上的那片白會跟白紙連成一塊。
在原始角度上找到的四角用 `_rotate_quad` 跟著圖片轉回擺正後的畫布。

用法上層只碰這裡：

    result, info = scan_auto(img)          # 自動挑
    result, info = scan(img, align_method='geometry', straighten_method='contrast')
"""

import cv2
import numpy as np

from auto_align_rotate import auto_align_rotate
from detect_angle_rect_only import (angle_matrix, apply_angle,
                                             detect_angle, detect_quad)
from detect_lines_and_rect_cv import detect_rect
from filter_out_background import fill_document_by_color

# ---------------------------------------------------------------- 設定

# 拉平後要輸出的頁面大小 (寬, 高)；None = 保持文件原本比例，不裁切
PAGE_SIZE = (1240, 1754)          # A4 300dpi

# 怎麼湊成這一頁。兩種都有代價，預設選不傷內容的那個：
#   'contain' 縮到一定放得下，多出來的地方補白邊 —— 內容完整，會留白邊
#   'cover'   放大到一定滿版，超出頁面的部分中心裁掉 —— 滿版，會切掉內容
# 收據這麼細長的文件（3.jpg 拉平後 201x530，比例 0.38）跟 A4（0.71）差很多，
# 用 'cover' 會為了滿版把上下切掉將近一半，選 'contain' 才不會把內容弄丟。
FILL_MODE = "contain"

# 偵測用的固定大小 (寬, 高)。圖等比例縮進這個框，**只在偵測時用**，
# 旋轉與輸出一律做在原圖上。設 None 就在原圖上偵測（慢，但不做任何假設）。
#
# 為什麼值得縮：線條偵測沿著掃描線生長，成本跟像素數成正比。
# 實測五張圖合計 86.72 秒 → 21.33 秒（4 倍），5.jpg 一張 76.83 秒 → 11.74 秒（6.5 倍），
# 而且四個方法的答案都沒有變。
#
# 縮多小還安全？實測 1240x1754 / 1000x1414 / 800x1131 / 600x849 四級，
# 找出來的四角誤差都是 0.0 像素。**但這個 0.0 是有水分的**：
# 這批圖裡 1/2/3/4.jpg 根本比 1240x1754 還小，從來沒被縮過；
# 唯一真的被縮到的大圖 5.jpg 不管縮不縮都找不到四角。
# 所以實測證明的是「縮了不會壞」，還沒證明「縮了會變好」。
# 要更小的話自己再量一輪。
WORK_SIZE = (1240, 1754)

# 偵測前要不要先轉黑白。'none' | 'otsu' | 'mean'
#
# 實測下來這件事**不能一視同仁**，每個方法適合的版本不一樣，所以
# 真正決定用哪個版本的是下面的 METHOD_VIEW，不是這個開關：
#
#   lines     一定要灰階。轉黑白之後 3.jpg 抓到的框變成 (5,3)(408,3)(408,734)(5,734)，
#             也就是整張畫面——黑白化在文件的輪廓外圍畫出一圈連續的階梯邊，
#             線條偵測會鎖死在那一圈上，而不是真正的紙邊。4.jpg 直接抓不到。
#             而且慢 2～3 倍（3.jpg 0.36s → 1.53s）。
#   geometry  沒差。角度一模一樣，5.jpg 還快一點（0.08s → 0.02s）。
#   contrast  沒差。它內部本來就做 Otsu，外面先轉等於做兩次一樣的事。
#   text      不能用，Tesseract 要的是灰階，不是二值。
BINARY_MODE = 'none'

# 每個方法要用哪個版本：'gray' 灰階 / 'binary' 二值。上面的註解說明為什麼。
# 兩個版本都只拿來「偵測」，輸出的圖永遠是原圖轉出來的。
# color 沒有列在上面的理由表裡：它的想法本來就是「先轉灰階再統計主色」，
# 灰階是它的輸入不是選項。給灰階版剛好符合它的預設路徑。
METHOD_VIEW = {'geometry': 'binary', 'text': 'gray',
               'lines': 'gray', 'contrast': 'binary', 'color': 'gray'}

# lines 方法找四角的嘗試順序：先用預設門檻，抓不到才換放寬的。
# 每多一組就多跑一次完整的線條偵測（四個方向各掃一次），大圖一組要 7 秒以上。
#   1. 預設：邊緣銳利、對比夠的圖（3.jpg）
#   2. 只把邊緣門檻放軟，線的長度要求不動（4.jpg）
LINE_TRIES = ({}, dict(pct=75))


# ================================================================= 小工具
def _shrink(img, box):
    """等比例縮到塞進 box，外框不變形。本來就比 box 小就不縮。"""
    if box is None:
        return img
    h, w = img.shape[:2]
    s = min(box[0] / float(w), box[1] / float(h))
    if s >= 1.0:
        return img
    return cv2.resize(img, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                      interpolation=cv2.INTER_AREA)


def _binarize(img, mode):
    """把影像轉成黑白。'none' 原樣回去。

    `otsu`      全域 Otsu，讓資料自己定門檻；適合一亮一暗乾乾淨淨的圖。
    `mean`      局部平均，適合有漸層陰影的照片（全域門檻會把陰影那半邊整片判成黑）。
    """
    if mode == 'none':
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if mode == 'otsu':
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if mode == 'mean':
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 25, 12)
    raise ValueError(f"沒有的黑白模式 {mode!r}，可用：'none'、'otsu'、'mean'")


def _turn(img, angle, turn):
    """照量到的角度旋轉，並視需要上下翻轉。兩個旋轉函式做的事一樣。"""
    out = apply_angle(img, angle)
    return cv2.rotate(out, cv2.ROTATE_180) if turn else out


def _rotate_quad(quad, shape, angle, turn):
    """把四個角跟著圖片一起轉到「旋轉後的畫布」上。

    用的是 `apply_angle` 同一份旋轉矩陣（`angle_matrix`），所以轉完的角
    就是旋轉後那張圖上的角——實測標記點的位置跟 M @ p 完全相符。

    `turn` 是上下翻轉（180°），跟 `_turn` 一樣，翻的是旋轉後畫布的中心：
    (x, y) → (nw−1−x, nh−1−y)。
    """
    pts = np.asarray(quad, np.float64).reshape(-1, 2)
    if angle:
        M, (nw, nh) = angle_matrix(shape, angle)
        out = pts @ M[:, :2].T + M[:, 2]
    else:
        h, w = shape[:2]
        nw, nh = w, h
        out = pts.copy()
    if turn:
        out = np.column_stack([nw - 1.0 - out[:, 0], nh - 1.0 - out[:, 1]])
    return out


def _warp(img, quad):
    """把四角拉平成矩形。寬高取兩條長邊的最大值，避免留黑邊。

    邊界補白，跟 apply_angle 用同一個理由：這裡露出的面積很小
    （只有四角四捨五入差的那一點點），但 BORDER_REPLICATE 拉出來的
    是文件自己的邊緣顏色，在深色文件上會是一圈更深的細邊，
    而 `_fill` 的 boundingRect 又會把它當成「內容」。
    """
    tl, tr, br, bl = quad
    width = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    height = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    width, height = max(width, 1), max(height, 1)
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                   np.float32)
    M = cv2.getPerspectiveTransform(np.asarray(quad, np.float32), dst)
    # borderValue 一定要給三元組：純量 255 會被當成 Scalar(255,0,0)，在 BGR 上是藍色。
    return cv2.warpPerspective(img, M, (width, height), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(255, 255, 255))


def _fill(img, size=PAGE_SIZE, mode=FILL_MODE):
    """先裁掉四角殘留的背景（不留黑邊），再湊成固定頁面大小。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(gray, 8, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    x, y, bw, bh = cv2.boundingRect(mask)
    if bw > 10 and bh > 10:
        img = img[y:y + bh, x:x + bw]

    if size is None:
        return img

    h, w = img.shape[:2]
    if mode == "cover":
        scale = max(size[0] / w, size[1] / h)
        big = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        rh, rw = big.shape[:2]
        x0, y0 = (rw - size[0]) // 2, (rh - size[1]) // 2
        return big[y0:y0 + size[1], x0:x0 + size[0]]

    scale = min(size[0] / w, size[1] / h)
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    canvas = np.full((size[1], size[0], 3), 255, np.uint8)
    sh, sw = small.shape[:2]
    x0, y0 = (size[0] - sw) // 2, (size[1] - sh) // 2
    canvas[y0:y0 + sh, x0:x0 + sw] = small
    return canvas


def _views(img):
    """回傳 (灰階版, 二值版)。兩個都只拿來偵測，輸出永遠用原圖。"""
    gray = _binarize(img, 'none')
    return gray, _binarize(img, BINARY_MODE)


def _view(img, name):
    """取出某個方法該用的那一個版本。"""
    return _views(img)[1 if METHOD_VIEW.get(name) == 'binary' else 0]


# ================================================================= 對齊旋轉
def align_geometry(img):
    """只用邊緣的角度擺正。快、免 Tesseract，但角度只確定到模 90。

    只回報歪斜、不處理上下翻轉——因為邊緣分不出「正拍」跟「上下顛倒」。
    """
    angle = detect_angle(_view(img, 'geometry'))
    return _turn(img, angle, 0), {'skew': angle, 'turn': 0,
                                  'note': f'邊緣角度 {angle:+.2f}°（模 90，不判翻轉）'}


def align_text(img):
    """用墨水的投影輪廓 + 認字擺正。角度與上下翻轉都處理。"""
    out, skew, turn, note = auto_align_rotate(_view(img, 'text'))
    return out, {'skew': skew, 'turn': turn, 'note': note}


ALIGN_METHODS = {
    'geometry': align_geometry,
    'text': align_text,
}


# ================================================================= 拉平整頁
def straighten_lines(img, tries=LINE_TRIES):
    """用「長線的交點」找四角。找不到回傳 None。

    一定要給灰階版（見 METHOD_VIEW 的說明）：黑白化會讓它抓到整張畫面。
    """
    view = _view(img, 'lines')
    which = -1
    quad = None
    for i, kwargs in enumerate(tries):
        quad = detect_rect(view, **kwargs)
        if quad is not None:
            which = i
            break
    if quad is None:
        return None, {'quad': None, 'try': -1,
                      'note': f'{len(tries)} 組門檻都找不到線條與四角'}
    tag = '預設門檻' if which == 0 else f'第 {which + 1} 組（放寬）'
    pts = ' '.join(f'({p[0]:.0f},{p[1]:.0f})' for p in quad)
    return quad, {'quad': quad, 'try': which, 'note': f'線條交點[{tag}] {pts}'}


def straighten_contrast(img):
    """用「文件跟背景的明暗分界」找四角。找不到回傳 None。"""
    quad = detect_quad(_view(img, 'contrast'))
    if quad is None:
        return None, {'quad': None, 'try': -1,
                      'note': '明暗分界找不到乾淨的四角（紙跟背景分不開，或文件就是整張畫面）'}
    return quad, {'quad': quad, 'try': 0,
                  'note': '明暗分界 ' + ' '.join(f'({p[0]:.0f},{p[1]:.0f})' for p in quad)}


def straighten_color(img, **kwargs):
    """用「門檻掃描 + 邊界線」找四角，並用顏色統計驗證，順便填出內白外黑的遮罩。

    模組裡原本的「主色／次色上色」那一版涵蓋率 0/5（實測見 filter_out_background.md 第五節），
    現在換成 `detect_document_by_color`：每一張圖自己在亮度直方圖上找門檻，
    提出候選四角，再用「框裡跟框外一圈是不是兩種顏色」驗證，最後把角收準到真正的邊界。

    實測涵蓋率 **2/5**（4.jpg、5.jpg），而且抓到的**跟 `lines` 不是同一批**：
    5.jpg 那張歪 45 度、還有摺痕的發票，`lines` 跟 `contrast` 都找不到，只有它找得到。
    1.jpg、2.png（收據本身就佔滿整張畫面）跟 3.jpg（白紙壓白桌）誠實地回傳 None。

    **要給它原始角度的圖，不要給擺正後的圖**（理由見 `NEEDS_ALIGNED`）。
    """
    quad, cinfo = fill_document_by_color(img, **kwargs)
    if quad is None:
        return None, {'quad': None, 'try': -1, 'note': cinfo['note']}
    pts = ' '.join(f'({p[0]:.0f},{p[1]:.0f})' for p in quad)
    return quad, {'quad': quad, 'try': 0, 'mask': cinfo.get('mask'),
                  'note': f'{cinfo.get("source", "四角")}{pts}'}


STRAIGHTEN_METHODS = {
    'lines': straighten_lines,
    'contrast': straighten_contrast,
    'color': straighten_color,
}

# scan_auto 自動嘗試的順序。lines 排前面是因為量過：它對 3.jpg、4.jpg 都對，
# contrast 只對 4.jpg 對，而且兩張的角點中心誤差都是 5 像素，
# 但 lines 在 3.jpg 上唯一能給出真正的收據輪廓。
#
# color 放最後是因為它抓到的是**另外一張**：5.jpg 那張歪 40 度、還有摺痕的發票，
# lines 跟 contrast 都找不到，只有 color 找得到，而且框得相當準。
# 放最後是因為它比 lines 慢（5.jpg 0.33 秒 vs lines 11.7 秒），
# 前面兩種找到就不用輪它。
STRAIGHTEN_ORDER = ('lines', 'contrast', 'color')

# 每一種拉平方法要在哪一張圖上跑。
#
# True  = 擺正（旋轉）之後的圖
# False = 原始角度的圖
#
# 分界線是「**靠亮度分出文件跟背景**」還是「**靠形狀找邊界**」：
#
# color、contrast 靠亮度分，**不能用擺正後的圖**。旋轉會把畫布撐大、
# 四角補白，那片白對白紙來說是致命的：
#   1. 分色時白紙跟補白連成一塊，門檻掃描根本分不出紙跟背景；
#   2. 「框外一圈的主色」量到的是補白而不是真正的背景，
#      水泥地看起來就像被白紙包住，於是框到水泥地上去了。
# 實測 5.jpg（擺正 45.5°）：在擺正後的圖上跑 color，找到的是水泥地；
# 同一張圖在原始角度上跑，找到的是發票本身。
# 這兩個方法本來就與角度無關，給它們原始角度的圖不會變差。
#
# lines 靠形狀找邊界，**一定要擺正後的圖**：它沿著掃描線找長邊，
# 紙歪的時候掃描線不會沿著紙邊走。
#
# 實測（本專案五張圖）把 contrast 改到原始角度跑，答案跟擺正後完全一樣
# （4.jpg 同樣是 (45,0)(196,0)(195,418)(44,418)，其餘四張同樣找不到），
# 純粹是為了不要踩到補白這個坑。
NEEDS_ALIGNED = {'lines': True, 'contrast': False, 'color': False}


# ================================================================= 介面
def align(img, method='text'):
    """用指定的方法對齊旋轉。回傳 (擺正後的圖, info)。"""
    if method not in ALIGN_METHODS:
        raise KeyError(f'沒有的對齊方法 {method!r}，可用：{list(ALIGN_METHODS)}')
    return ALIGN_METHODS[method](img)


def straighten(img, method='lines', size=PAGE_SIZE, mode=FILL_MODE, **kwargs):
    """用指定的方法找四角、拉平、湊頁面。回傳 (湊好頁面的圖 或 None, info)。

    單獨呼叫這個 = 完整跑一次那個方法，不經過縮圖流程。
    """
    if method not in STRAIGHTEN_METHODS:
        raise KeyError(f'沒有的拉平方法 {method!r}，可用：{list(STRAIGHTEN_METHODS)}')
    quad, info = STRAIGHTEN_METHODS[method](img, **kwargs)
    if quad is None:
        return None, info
    return _fill(_warp(img, quad), size, mode), info


def _joined(ainfo, sinfo, found):
    """把兩層的資訊合併。

    兩層的函式各自都有 `note`，合併時直接展開會讓拉平的說明蓋掉擺正的說明，
    所以合併後一律改用 `align_note` / `straighten_note`。
    """
    return {**ainfo, **sinfo, 'found': found,
            'align_note': ainfo['note'], 'straighten_note': sinfo['note']}


def _pipeline(img, align_method, order, tries_by_name, box, size, mode):
    """縮圖 → 在小圖上量角度與四角 → 乘回原圖 → 在原圖上轉、裁、排。

    回傳 (頁面 或 擺正後的原圖, info)。
    """
    small = _shrink(img, box)
    flat, ainfo = align(small, align_method)

    # 原圖做一模一樣的旋轉。角度是無號的可以直接用；
    # 「一模一樣」是對應座標的前提——小圖跟原圖差剛好一個縮放，
    # 所以小圖上量到的四角只要乘上尺寸比就是原圖的座標。
    full = _turn(img, ainfo['skew'], ainfo['turn'])
    sx = full.shape[1] / float(flat.shape[1])
    sy = full.shape[0] / float(flat.shape[0])

    tried = []
    for name in order:
        # 有些方法要在擺正後的圖上跑（lines），有些要在原始角度的圖上跑（color，
        # 理由見 NEEDS_ALIGNED）。在原始角度上找到的角，要跟著圖片轉到
        # 擺正後的畫布上才對得起來。
        aligned = NEEDS_ALIGNED.get(name, True)
        view = flat if aligned else small
        quad, sinfo = STRAIGHTEN_METHODS[name](view, **tries_by_name.get(name, {}))
        tried.append(f'{name}: {sinfo["note"]}')
        if quad is not None:
            if not aligned:
                quad = _rotate_quad(quad, small.shape[:2],
                                    ainfo['skew'], ainfo['turn'])
            back = np.asarray(quad, np.float64) * np.array([sx, sy])
            info = _joined(ainfo, {**sinfo, 'quad': back}, True)
            info.update(method=name, tried=tried, shrink=box)
            return _fill(_warp(full, back), size, mode), info
    sinfo = {'quad': None, 'try': -1,
             'note': f'{len(order)} 種拉平方法都找不到四角：' + ' / '.join(tried)}
    info = _joined(ainfo, sinfo, False)
    info.update(method=None, tried=tried, shrink=box)
    return full, info


def scan(img, align_method='text', straighten_method='lines', box=WORK_SIZE,
         size=PAGE_SIZE, mode=FILL_MODE, **kwargs):
    """先對齊再用指定的那一種拉平方法。回傳 (頁面, info)。

    找不到四角時回傳「對齊後的原圖」並標記 found=False——硬把整張照片
    塞進 A4 只是多加一圈白邊，不會讓文件變清楚。
    """
    if straighten_method not in STRAIGHTEN_METHODS:
        raise KeyError(f'沒有的拉平方法 {straighten_method!r}，'
                       f'可用：{list(STRAIGHTEN_METHODS)}')
    return _pipeline(img, align_method, (straighten_method,),
                     {'lines': dict(tries=LINE_TRIES)}, box, size, mode)


def scan_auto(img, align_method='text', order=STRAIGHTEN_ORDER, box=WORK_SIZE,
              size=PAGE_SIZE, mode=FILL_MODE, **kwargs):
    """對齊之後，依次試各種拉平方法，用第一個成功的。

    回傳 (頁面, info)：`method` 是實際成功的方法，全部失敗則為 None；
    `tried` 是每個方法的結果，不論有沒有成功都會列出，方便回頭查原因。
    """
    return _pipeline(img, align_method, order,
                     {'lines': dict(tries=LINE_TRIES)}, box, size, mode)


def list_methods():
    """回傳方法清單的字串表格。"""
    rows = [('組別', '名稱', '用哪個版本', '說明')]
    view_cn = {'gray': '灰階', 'binary': '二值'}
    for grp, table in (('對齊', ALIGN_METHODS), ('拉平', STRAIGHTEN_METHODS)):
        for name, fn in table.items():
            rows.append((grp, name, view_cn[METHOD_VIEW[name]],
                         fn.__doc__.strip().split('\n')[0]))
    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    return '\n'.join('  '.join(r[i].ljust(widths[i]) for i in range(4)) for r in rows)


if __name__ == '__main__':
    import os
    import sys
    import time

    sys.stdout.reconfigure(encoding='utf-8')
    print(list_methods())
    print(f'\n偵測用大小 {WORK_SIZE}，黑白模式 {BINARY_MODE}\n')
    src = sys.argv[1] if len(sys.argv) > 1 else 'src'
    os.makedirs('out', exist_ok=True)
    for name in sorted(os.listdir(src)):
        if os.path.splitext(name)[1].lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
            continue
        im = cv2.imread(os.path.join(src, name))
        if im is None:
            continue
        t = time.time()
        page, info = scan_auto(im)
        dst = os.path.join('out', os.path.splitext(name)[0] + '_scan.jpg')
        cv2.imwrite(dst, page)
        print(f'{name:10s} {time.time() - t:6.2f}s  拉平用 {info["method"] or "（沒有）"}')
        print(f'{"":10s} 擺正 {info["align_note"]}')
        for line in info['tried']:
            print(f'{"":10s}   {line}')