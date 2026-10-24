"""自動對齊與旋轉：判斷一張文件照片該轉幾度，並把傾斜拉正。

分成兩件互相獨立的事：

1. **任意角度的歪斜**（skew）— 紙拍歪了幾度，跟轉向無關。
   靠「投影輪廓的變異數」量出來，範圍 −60°～+60°，不只修 ±1° 這種小歪。
2. **90° 的倍數**（turn）— 紙是正拍還是橫拍、上還是正下。這個量不出來
   （一張正立的 A4 轉 180° 之後所有水平線還是水平線），只能認字。

為什麼先擺正再判斷轉向：認字是靠「一行一行的字」認的，字歪 45° 的時候
連一行都切不出來。5.jpg 擺正之前 Tesseract 直接回報「字太少」放棄判斷，
擺正之後認字就有 159 個詞了。

轉向是比較「0 度」和「180 度」兩張圖認字的**平均信心度**決定的，
沒有用 Tesseract OSD：實測 OSD 餵十次答錯四次，而且信心度高低跟對錯無關
（1.jpg 本來就是正的，OSD 卻說要翻轉，還給了 1.05 的信心度；
4.jpg 的正解它給的信心度只有 0.17，低到任何門檻都會把對的答案丟掉）。
細節在 auto_align_rotate.md。

做法與實測數據的說明在 auto_align_rotate.md。
"""

import os
import re

import cv2
import numpy as np

try:
    from detect_angle_rect_only import angle_matrix
except ImportError:
    # 直接執行這個檔案（python jCVTools/auto_align_rotate.py）的時候，
    # 套件路徑不在 sys.path 上，改用同資料夾的版本。
    from detect_angle_rect_only import angle_matrix

# Tesseract 不是 Python 套件，是獨立安裝的執行檔，這個路徑要自己指給它
TESSERACT_DIR = r'C:\Program Files\Tesseract-OCR'
TESSDATA_PREFIX = os.path.join(TESSERACT_DIR, 'tessdata')

# 認字時影像短邊至少要做到這麼大。太小 Tesseract 會回報解析度 0 dpi，
# 認出來的字一堆亂碼，平均信心度會被拉低到沒有區別力。
OCR_MIN_SIDE = 1000

# 但也不能超過這個大小。認字的品質在 1800 像素左右就飽和了，
# 再大只是變慢：5.jpg 擺正後有 2882×2883，縮到 1800 之後兩個方向認字
# 從 2.9 秒變 0.66 秒，而且正立 63.9 / 倒立 22.7 的分離度比原尺寸的
# 61.6 / 27.3 還要乾淨。
#
# 這個上限不能設太小。試過縮到 1000：5.jpg 倒立方向的信心度直接掉到 0.0，
# 那是「一個字都認不出來」而不是「認得很差」。那種差距是假的——
# 一張真的上下顛倒但清楚的紙，縮到同樣的大小，正立方向也會認不出字，
# 兩邊一起變 0，差距就沒了，最後會判成維持原樣（也就是漏翻）。
OCR_MAX_SIDE = 1800

# 正立方向的信心度超過這個值就不用再認 180 度那次。
# 實測五張圖，正立方向是 52～86，倒立方向最高只有 33，中間是空的，
# 所以 70 是個很安全的門檻。清楚的大文件通常一次就判完，省掉半秒。
CONFIDENT_CONF = 70.0

# 一行字裡至少要有這麼多個字母數字才算一個「詞」
_WORD = re.compile(r'[0-9A-Za-z]{2,}')


# ---------------------------------------------------------------- 1. 量傾斜角度
def _ink_mask(img, target_w=900, block=25, c=12):
    """取出「有墨」的遮罩，並把寬度縮到 target_w 左右。

    用局部平均閾值而不是全域固定閾值：文件常有漸層陰影，
    固定閾值會把陰影那半邊整片判成黑色，輪廓會被一整塊墨吃掉。
    縮圖是為了讓下面兩百多次旋轉掃描跑得完（原圖掃一次要 0.1 秒以上）。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape[:2]
    scale = target_w / float(w)
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, block, c)
    return (bw > 0).astype(np.float32)


def _profile_variance(ink, angle):
    """旋轉 angle 之後，把每一列的墨水量加總，取這個向量的變異數。

    為什麼用變異數：文字排正時同一行的字都疊在同一列上，
    「有字的一列」墨很多、「行與行之間的一列」墨很少，高低差大，變異數大。
    紙轉歪的時候字被平均攤到所有列上，高低差被抹平，變異數就小。
    這個量完全不認字，所以歪到 45° 也量得出來（不像用字框量角度會失效）。
    """
    h, w = ink.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), float(angle), 1.0)
    rot = cv2.warpAffine(ink, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    return float(rot.sum(axis=1).var())


def estimate_skew(img, limit=60.0, coarse=2.0, mid=0.25, fine=0.05):
    """回傳讓文字排正所需要的旋轉角度（度，正值＝逆時針）。

    分三段逼近：先 2° 的步距掃過 −60°～+60°，再 0.25° 掃第一名附近 ±2°，
    最後 0.05° 掃 ±0.25°。

    為什麼不是「0.5° 掃全程、再 0.05° 細掃」：那是 262 次輪廓計算，
    分三段只要 89 次，五張圖從 0.703 秒降到 0.239 秒，
    **量出來的角完全不變**（−0.00、+0.40、−0.00、−0.00、+45.50，誤差 0.000 度）。
    輪廓變異數在幾度的範圍裡變化很慢，2° 的粗掃足以鎖定正確的山頭。

    最後要細到 0.05° 是因為 0.5° 對後續認字來說還是偏了：
    200dpi 的字高只有 10 像素左右，0.5° 的誤差到頁面下方就累積成好幾像素。

    回傳 (角度, 變異數)；墨量不足量不出來時變異數為 0.0。
    """
    ink = _ink_mask(img)

    def sweep(lo, hi, step):
        angles = np.arange(lo, hi + 1e-9, step)
        scores = [_profile_variance(ink, a) for a in angles]
        return float(angles[int(np.argmax(scores))]), float(max(scores))

    top, score = sweep(-limit, limit, coarse)
    if score <= 0.0:
        return 0.0, 0.0

    top, score = sweep(top - coarse, top + coarse, mid)
    top, score = sweep(top - mid, top + mid, fine)
    return top, score


def apply_skew(img, angle):
    """把影像轉 angle 度讓文字排正（angle 直接用 estimate_skew 回傳的值）。

    畫布會跟著放大。轉 45° 之後原本塞滿畫面的文件會超出四邊，
    只轉不放大等於把文件的角切掉——這在 5.jpg 這種歪很多的照片上很明顯。

    放大出來的四角補白。原先用的是 BORDER_REPLICATE（把邊緣像素拉出去填），
    那會在角落拖出一條條拉長的邊緣色帶，看起來像髒邊；
    BORDER_CONSTANT + 白才是乾淨的。
    """
    if not angle:
        return img
    M, (nw, nh) = angle_matrix(img.shape[:2], angle)
    # borderValue 一定要給三元組。實測：給純量 255，OpenCV 會當成
    # Scalar(255, 0, 0)，在 BGR 上就是**藍色**，不是白色。給 (255, 255, 255)
    # 才是白；灰階圖取第一個分量，一樣是 255，兩種圖都對。
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(255, 255, 255))


# ---------------------------------------------------------------- 2. 認字
def _setup_tesseract():
    """把 Tesseract 的位置寫進環境變數。回傳 False 表示用不了。

    這一步不能省。TESSDATA_PREFIX 是環境變數，同一台電腦上跑過別的專案
    （例如某個 .NET 測試專案把輸出目錄塞進去）之後，這裡就會讀到那份不完整的
    tessdata，結果是 UnicodeDecodeError 或 'Too few characters'，
    跟圖片本身完全無關，很難查。
    """
    exe = os.path.join(TESSERACT_DIR, 'tesseract.exe')
    if not os.path.exists(exe):
        return False
    try:
        import pytesseract
    except ImportError:
        return False

    os.environ['TESSDATA_PREFIX'] = TESSDATA_PREFIX
    os.environ['PATH'] = TESSERACT_DIR + os.pathsep + os.environ.get('PATH', '')
    pytesseract.pytesseract.tesseract_cmd = exe
    return True


def _ocr_ready(img, min_side=OCR_MIN_SIDE, max_side=OCR_MAX_SIDE):
    """把影像調整到 Tesseract 最好用的尺寸。

    太小要放大（否則回報解析度 0 dpi，認出來全是亂碼），
    太大要縮小（品質在 1800 像素就飽和，再大只是變慢）。
    """
    h, w = img.shape[:2]
    side = min(h, w)
    scale = min_side / float(side)
    if side * max(scale, 1.0) > max_side:
        scale = max_side / float(side)
    if abs(scale - 1.0) < 1e-3:
        return img
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=interp)


def ocr_score(img, ready=False):
    """回傳 (認出的詞數, 這些詞的平均信心度)。

    ready=True 表示 img 已經調過尺寸，可以直接認。
    連續認兩次時用它省掉一次縮圖——先縮再翻比先翻再縮便宜很多
    （5.jpg 翻轉一張 2882 像素的圖比重翻一張 1800 像素的圖貴四倍）。

    信心度是這裡真正的判別依據，不是詞數。實測上下翻轉的版本詞數幾乎不減
    （翻轉後的字形變形，仍然切得出等長的字母數字串），但信心度會從 50 以上
    掉到 30 以下——Tesseract 自己知道那些字認不認得出來。
    """
    from pytesseract import Output
    import pytesseract

    if not ready:
        img = _ocr_ready(img)
    data = pytesseract.image_to_data(img, output_type=Output.DICT,
                                     config='--oem 1 --psm 6')
    words = 0
    scored = 0
    total = 0.0
    for text, conf in zip(data['text'], data['conf']):
        text = (text or '').strip()
        if not _WORD.match(text):
            continue
        words += 1
        try:                 # conf 可能是 -1 或空字串，那個詞認不出來，不列入平均
            total += float(conf)
            scored += 1
        except (TypeError, ValueError):
            pass
    return words, (total / scored if scored else 0.0)


# ---------------------------------------------------------------- 3. 判斷 0 或 180
def estimate_turn(img, min_margin=10.0, confident=CONFIDENT_CONF):
    """決定要不要上下翻轉。回傳 (角度, 說明)。

    只有 0 和 180 兩個候選。這是因為呼叫它之前一定已經擺正過歪斜，
    文字已經是橫的了，剩下的就只有「是不是上下顛倒」這一個問題。
    （橫拍 90 / 270 的情況不在這裡處理，那要靠 OSD，而且照樣不可靠。）

    正立方向認得很清楚的時候（信心度超過 confident）就不再認第二次，
    因為倒立方向再清楚也就 30 出頭，中間是一段空的，沒有翻轉的可能。

    min_margin 是「兩個方向的信心度差多少才敢決定」。差太少代表這張圖
    字太少或太糊，兩個方向認起來差不多，寧可回傳 0 也不要亂猜。
    實測五張測試圖的差距是 26～52，最低門檻 10 有很大餘裕。
    """
    if not _setup_tesseract():
        return 0, 'Tesseract 不可用，維持原樣'

    try:
        small = _ocr_ready(img)
        w0, c0 = ocr_score(small, ready=True)
        if c0 >= confident:
            return 0, f'0 度：{w0} 詞 / 信心度 {c0:.1f}，夠清楚，不必再試 180 度 → 維持 0 度'
        w180, c180 = ocr_score(np.ascontiguousarray(np.rot90(small, 2)), ready=True)
    except Exception as exc:
        return 0, f'認字失敗（{type(exc).__name__}），維持原樣'

    detail = f'0 度：{w0} 詞 / 信心度 {c0:.1f}；180 度：{w180} 詞 / 信心度 {c180:.1f}'
    if abs(c0 - c180) < min_margin:
        return 0, f'{detail}；差距不到 {min_margin}，判定不出來，維持原樣'
    if c0 > c180:
        return 0, f'{detail} → 維持 0 度'
    return 180, f'{detail} → 翻轉 180 度'


# ---------------------------------------------------------------- 對外主流程
def auto_align_rotate(img, min_margin=10.0, limit=60.0):
    """回傳 (轉正後的影像, 傾斜角, 轉向角, 說明字串)。

    傾斜角為 0.0 表示量不到（墨量不足），圖片不會被旋轉。
    轉向角一定是 0 或 180；判定不出來時回傳 0，說明字串裡會寫清楚。
    """
    skew, score = estimate_skew(img, limit=limit)
    flat = apply_skew(img, skew)
    note = (f'擺正歪斜 {skew:+.2f}°（輪廓變異數 {score:.0f}）'
            if score > 0.0 else '墨量不足，量不到傾斜，保持原角度')

    turn, why = estimate_turn(flat, min_margin=min_margin)
    if turn:
        flat = np.rot90(flat, 2).copy()
    return flat, skew, turn, f'{note}；{why}'


if __name__ == '__main__':
    import sys

    sys.stdout.reconfigure(encoding='utf-8')     # Windows 主控台預設不是 utf-8，中文會變亂碼
    src = sys.argv[1] if len(sys.argv) > 1 else 'src'
    os.makedirs('out', exist_ok=True)

    for name in sorted(os.listdir(src)):
        if os.path.splitext(name)[1].lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
            continue
        im = cv2.imread(os.path.join(src, name))
        if im is None:
            continue
        out, skew, turn, note = auto_align_rotate(im)
        stem = os.path.splitext(name)[0]
        dst = os.path.join('out', f'{stem}_aligned.jpg')
        cv2.imwrite(dst, out)
        print(f'{name:10s} {im.shape[1]}x{im.shape[0]} -> {out.shape[1]}x{out.shape[0]}')
        print(f'{"":10s} {note}')
        print(f'{"":10s} 輸出 {dst}')