# JOpenCV — 文件照片的對齊與拉平

把一張拍歪的文件照片，變成一頁端正的 A4。

```
原始照片  →  擺正（轉幾度、要不要翻）  →  找四角  →  拉平（透視校正）  →  湊成頁面
```

專案裡有**兩組各自獨立**的方法可以完成上面每一件事，
上層用介面把它們對應到名字，可以單獨用、也可以自動挑。
每一種方法認的東西都不一樣（邊緣線的角度、墨水的輪廓、文件的輪廓、明暗分界、顏色統計），
所以它們**在不同照片上各自有輸贏**——這正是把介面抽出來的理由。

---

## 快速開始

```bash
python main.py                       # 批次處理 src/ 裡的照片，結果寫到 out/
python -m methods src       # 同上，但會印出每個方法的判斷過程
```

```python
from methods import scan_auto

page, info = scan_auto(img)          # 自動挑一個成功的方法
print(info['align_note'], info['method'], info['found'])
```

`page` 是湊好頁面的圖；找不到四角時回傳擺正後的原圖，
並且 `info['found']` 是 `False`、`info['method']` 是 `None`。

---

## 五個方法

### 對齊旋轉（決定照片歪了幾度、該不該翻轉）

| 名稱 | 認什麼 | 優點 | 缺點 | 模組 |
|------|--------|------|------|------|
| `geometry` | 邊緣線的角度（霍夫） | 快、免 Tesseract | 角度只確定到模 90 | `detect_angle_rect_only` |
| `text` | 墨水的投影輪廓 + 認字 | 連模 90 的正負號、上下翻轉都處理 | 慢、要 Tesseract | `auto_align_rotate` |

### 拉平整頁（找出文件的四角、拉平、湊成頁面）

| 名稱 | 認什麼 | 優點 | 缺點 | 模組 |
|------|--------|------|------|------|
| `lines` | 沿掃描線長出來的邊界線 | 紙邊斷掉還是找得到 | 慢、對邊緣對比弱的紙無解 | `detect_lines_and_rect_cv` |
| `contrast` | 文件跟背景的明暗分界 | 極快 | 紙跟背景亮度接近就沒用 | `detect_angle_rect_only` |
| `color` | 門檻掃描 + 邊界線，再用顏色統計驗證 | 認的是顏色不是形狀，快兩個數量級 | 紙跟背景顏色接近就沒用 | `filter_out_background` |

`scan_auto` 的預設嘗試順序是 `('lines', 'contrast', 'color')`。

---

## 實測

`src/` 這五張照片跑完整流程（`align_method='text'`）：

| 圖 | 尺寸 | 拉平用哪個 | 輸出 | 說明 |
|----|------|-----------|------|------|
| 1.jpg | 340×588 | — | 擺正後的原圖 | 收據佔滿整張畫面，沒有背景可以分 |
| 2.png | 1200×1200 | — | 擺正後的原圖 | 同上 |
| 3.jpg | 415×739 | `lines` | 1240×1754 | 白紙壓白桌，只有邊界幾何分得出來 |
| 4.jpg | 236×419 | `lines` | 1240×1754 | 白紙壓黑桌 |
| 5.jpg | 2074×2002 | **`color`** | 1240×1754 | 歪 45 度、有摺痕，`lines` 跟 `contrast` 都找不到 |

**涵蓋率 3/5。** 找不到的那兩張是誠實地回傳 `None`，
不是回傳一個看起來對但其實錯的框。

| 方法 | 找得到 | 哪些 |
|------|--------|------|
| `lines` | 2/5 | 3.jpg、4.jpg |
| `contrast` | 1/5 | 4.jpg |
| `color` | 2/5 | 4.jpg、5.jpg |

`color` 抓到的是**另外一張**（5.jpg），所以三種合起來才會有 3/5。

耗時：整批五張圖，原尺寸 82.93 秒，縮到 `WORK_SIZE = (1240, 1754)` 之後 21.75 秒。

---

## 幾個不顯而易見的決定

這些是量出來的，不是拍腦袋定的；細節在各自的 `.md` 裡。

**縮圖偵測、原圖輸出。** 量角度、找四角都在縮圖上做，旋轉和輸出都在原圖上，
所以輸出解析度不會因為縮圖而損失。實測五張合計 82.93 秒 → 21.75 秒（3.8 倍），
而且答案一模一樣。

**靠亮度分文件跟背景的方法，不能跑在擺正後的圖上。** 旋轉會把畫布撐大、四角補白，
那片白對白紙來說是致命的：分色時白紙跟補白連成一塊，
而「框外一圈的主色」量到的是補白而不是真正的背景——
水泥地看起來就像被白紙包住，於是框到水泥地上去了。
實測 5.jpg（擺正 45.5°）：在擺正後的圖上跑 `color` 找到的是水泥地，
同一張圖在原始角度上跑，找到的是發票本身。
所以 `NEEDS_ALIGNED` 讓 `contrast`、`color` 拿**原始角度**的圖，
在原始角度上找到的四角再用同一份旋轉矩陣轉回擺正後的畫布。

**寧可回傳 `None`，不要硬掰一個框。** 一個亂猜的四角會讓輸出看起來
「有處理過」但其實裁錯，比直接承認做不到更麻煩。
`info['tried']` 會列出每個方法被刷掉的理由。

**每張圖自己找門檻。** 紙跟背景差多少灰階，每張圖都不一樣，
所以門檻不能寫死成一個共用的容差——那正是舊版 `color` 涵蓋率 0/5 的原因。

**文件歪 45 度時不能靠「x+y 最小 = 左上」排角。** 四個角在 x+y 跟 x−y 上會打成平手，
會排出錯的順序，拉出一個破掉的矩形。改成繞中心點用角度排成順時針一圈。

---

## 專案結構

```
main.py                            上層只呼叫 methods
jCVTools/
  methods.py                       介面層：名字 ↔ 函式、統一回傳格式、縮圖偵測、自動挑方法
  auto_align_rotate.py             對齊 text：投影輪廓 + 認字
  detect_angle_rect_only.py        對齊 geometry、拉平 contrast：只看邊緣
  detect_lines_and_rect_cv.py      拉平 lines：沿掃描線找長邊
  filter_out_background.py         拉平 color：門檻掃描 + 顏色統計驗證
src/                               測試照片
out/                               輸出
```

每個模組都有對應的 `.md`，記錄它的做法、參數與**實測數據**：

| 文件 | 記錄什麼 |
|------|---------|
| [`jCVTools/methods.md`](jCVTools/methods.md) | 介面本身：怎麼對應名字、怎麼換算座標、自動挑選怎麼決定 |
| [`jCVTools/filter_out_background.md`](jCVTools/filter_out_background.md) | `color`：門檻掃描、顏色驗證、收準四角，以及舊版為什麼 0/5 |
| [`jCVTools/detect_lines_and_rect_cv.md`](jCVTools/detect_lines_and_rect_cv.md) | `lines`：怎麼找出紙張的邊界線並接成四個角 |
| [`jCVTools/detect_angle_rect_only.md`](jCVTools/detect_angle_rect_only.md) | `geometry` 與 `contrast`：只看邊緣能做的事與不能做的事 |
| [`jCVTools/auto_align_rotate.md`](jCVTools/auto_align_rotate.md) | `text`：歪斜與轉向是兩個問題，分開處理 |

---

## 需求

- Python 3.9 以上（實測環境 3.12.3）
- **Tesseract**（只有對齊的 `text` 方法需要，用來認字判斷上下翻轉；
  沒裝的話歪斜還是可以修，但翻轉會維持原樣）

```bash
pip install -r requirements.txt
```

Tesseract 是獨立執行程式，不是 pip 套件，要另外裝，
並且把執行檔的路徑寫在 `jCVTools/auto_align_rotate.py` 開頭的 `TESSERACT_DIR`。

---

## 授權

還沒有選授權。發布到 GitHub 前請自己挑一個（MIT / Apache-2.0 / …）
並加一個 `LICENSE` 檔案。
`src/` 裡的照片是這個專案拿來測試用的素材。
