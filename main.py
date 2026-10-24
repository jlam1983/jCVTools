"""把 src 資料夾裡的照片拉平成頁面。

這支檔案只做一件事：呼叫 `methods`，把照片交給它處理。
實際的演算法都在底下的模組裡，各方法彼此獨立、也可以各自直接呼叫：

| 對齊旋轉 | 認什麼 | 模組 |
|----------|--------|------|
| `geometry` | 邊緣線的角度（霍夫） | `detect_angle_rect_only` |
| `text` | 墨水的投影輪廓 + 認字 | `auto_align_rotate` |

| 拉平整頁 | 認什麼 | 模組 |
|----------|--------|------|
| `lines` | 沿掃描線長出來的邊界線 | `detect_lines_and_rect_cv` |
| `contrast` | 文件跟背景的明暗分界 | `detect_angle_rect_only` |
| `color` | 門檻掃描＋邊界線，再用顏色統計驗證四角 | `filter_out_background` |

`color` 在預設嘗試順序裡（最後一個）：它抓到的是**另外一張**——
5.jpg 那張歪 45 度、還有摺痕的發票，`lines` 跟 `contrast` 都找不到，只有它找得到。
所以整條流程的涵蓋率是 3/5（3.jpg、4.jpg 用 `lines`，5.jpg 用 `color`），
1.jpg、2.png 誠實地回傳 None（收據本身就佔滿整張畫面，沒有背景可以分）。
數據見 `filter_out_background.md`。

流程是「縮圖 → 擺正 → 找四角 → 拉平 → 湊頁面」。
縮圖只用在偵測：量角度、找四角都在縮圖上做，旋轉和輸出都在原圖上，
所以輸出解析度不會因為縮圖而損失。

預設用 `scan_auto`：擺正用 `text`，找四角先試 `lines`，再試 `contrast`，
最後試 `color`，三個都失敗就停在擺正後的圖片——硬把整張照片塞進 A4
只是多加一圈白邊，不會讓文件變清楚。

各方法的做法、參數與實測數據寫在各自的 .md，`methods.md` 記錄介面本身。
"""

import os
import sys

import cv2
import numpy as np
from pathlib import Path

from methods import scan_auto

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / 'src'
OUT_DIR = BASE_DIR / 'out'


def scan_file(path, show=False):
    """讀一張照片、擺正拉平、存檔。回傳 (擺正拉平後的圖, 過程資訊)。"""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)

    page, info = scan_auto(img)
    print(f"  擺正: {info['align_note']}")
    for line in info['tried']:
        print(f"  試 {line}")
    if info['found']:
        print(f"  湊成頁面: {page.shape[1]}x{page.shape[0]}")
    else:
        print("  三種方法都找不到四角，只輸出擺正後的圖片（不拉平、不湊頁面）")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, os.path.splitext(os.path.basename(path))[0] + "_scan.jpg")
    cv2.imwrite(out, page)
    print("  輸出:", out)

    if show:
        pw = 420
        before = cv2.resize(img, (pw, int(pw * img.shape[0] / img.shape[1])))
        after = cv2.resize(page, (pw, int(pw * page.shape[0] / page.shape[1])))
        cv2.imshow("Before / After", np.hstack([before, after]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return page, info


def main(show=False):
    sys.stdout.reconfigure(encoding='utf-8')   # Windows 主控台預設不是 utf-8，中文會變亂碼
    for name in sorted(os.listdir(SRC_DIR)):
        if os.path.splitext(name)[1].lower() in (".jpg", ".jpeg", ".png", ".bmp"):
            print(name)
            scan_file(os.path.join(SRC_DIR, name), show=show)


if __name__ == "__main__":
    main(show=False)