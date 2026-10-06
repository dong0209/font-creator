# font-creator：NC 字型產生器

上傳一張有文字的圖片，程式會：

1. **切出圖中的每個字**，並**分析文字特色**：筆畫粗細、粗細對比、傾斜角度、字寬比、邊角圓潤度、手寫感，推測字體類型（黑體／明體／圓體／手寫）與字重。
2. 讓你挑 1–3 個字當**風格參考**，用 AI 模型 [FontDiffuser](https://github.com/yeungchenwa/FontDiffuser) 以這個風格**生成整套中文字與標點**。
3. **英文與數字**改從電腦裡的字型中自動挑出風格最接近的一套，再套上圖中的傾斜與粗細。
4. 把所有字描成向量輪廓，輸出 **TrueType 字型檔（.ttf）**。

> **此程式產生的字型禁止用於商業用途。** 字型名稱一律自動加上「NC」（Non-Commercial）標記，例如輸入「MyHand」會得到「MyHand NC」。

## 品質最佳化做了什麼

| 環節 | 做法 |
|---|---|
| 內容字型 | 預設用 FontDiffuser 訓練時的**開心宋體 A**；每套字型都依「國」字大小正規化成訓練時的 92%，缺字時自動改用其他中文字型補上 |
| 風格參考圖 | 依訓練資料的比例（字佔畫布約 80%）置中、白底黑字 |
| 多參考字 | 選 2–3 個參考字時，每個字各生成一次，依「結構是否完整、粗細是否一致、有無雜點」自動挑最好的 |
| 描圖 | 放大 4 倍 → 去雜點、補小洞 → potrace 曲線擬合 → 轉成 TrueType 曲線 |
| 粗細一致 | 依每個字在內容字型中的筆畫粗細，把生成結果外擴或內縮到一致的比例（筆畫多的字仍維持較細） |
| 英數字 | 模型以中文訓練，英數字品質差，所以改用最接近的現成字型，並以距離場調整粗細、錯切調整傾斜 |

## 要跑多久

FontDiffuser 是擴散模型，每個字要跑 20 次（可調）神經網路推論；**選幾個參考字，時間就乘以幾倍**。英數字不經 AI，幾秒就好。

| 環境 | 每字（20 步、1 個參考字） | 常用字 5,401 字 |
|---|---|---|
| 4 核心雲端 CPU（實測） | 約 28 秒 | 約 42 小時 |
| Apple M4（MPS GPU，未實測） | 估計 2–5 秒 | 估計 3–8 小時 |
| NVIDIA 獨立顯示卡 | 估計 < 1 秒 | 估計 1–2 小時 |

- 網頁上的「試生成」會量出你電腦的實際速度，並換算整套字型需要多久。
- 步數改成 10 可以快一倍，品質會稍差。
- 生成可以隨時**暫停**，關掉程式也沒關係：已生成的字會保留，下次按「繼續」接著跑。
- 預設字集是**常用字 5,401 字**（Big5 常用字區，依教育部常用國字表編排），另有「常用＋次常用 13,053 字」與「自訂字表」可選。

## 安裝

需要 Python 3.10 以上與 git。

### macOS（含 M 系列晶片）

```bash
git clone https://github.com/dong0209/font-creator.git
cd font-creator
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/setup_fontdiffuser.py
```

### Windows（PowerShell）

```powershell
git clone https://github.com/dong0209/font-creator.git
cd font-creator
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/setup_fontdiffuser.py
```

有 NVIDIA 顯示卡的話，請先依 <https://pytorch.org/get-started/locally/> 安裝 CUDA 版 PyTorch，再執行 `pip install -r requirements.txt`。

### 還需要手動放的東西

`setup_fontdiffuser.py` 會下載 FontDiffuser 原始碼，並告訴你還缺什麼：

1. **模型權重（必要）**：從 [FontDiffuser 的 Google Drive](https://drive.google.com/drive/folders/12hfuZ9MQvXqcteNuz7JQ2B_mUcTr-5jZ) 下載 `unet.pth`、`content_encoder.pth`、`style_encoder.pth`，放到 `vendor/ckpt/`。
2. **開心宋體 A（強烈建議）**：網路搜尋「开心宋体A」或「KaiXinSongA」下載 `.ttf`，放到 `fonts/`。這是 FontDiffuser 訓練時用的內容字型，效果最好。沒有的話會改用電腦裡的宋體／明體。

再執行一次 `python scripts/setup_fontdiffuser.py` 確認全部就緒。

## 使用

```bash
python -m fontmaker.server
```

打開 <http://127.0.0.1:8765>（第一次啟動會花幾十秒掃描電腦裡的字型，之後有快取）：

1. **上傳圖片** → 自動切字並顯示「文字特色分析」。
2. **選風格參考字**（已依適合程度排序，筆畫多、方正、清楚的中文字最好；可多選 1–3 個）→ 確認內容字型（★ 為最推薦）與英數字字型 → 按「試生成」看效果。可以換參考字或隨機種子多比較幾次。
3. 滿意後按「用這組設定生成完整字型」，輸入字型名稱、選字集 → **開始生成**。
4. 完成後**下載 .ttf**，也可以直接在頁面上試打。

## 設定（環境變數）

| 變數 | 預設 | 說明 |
|---|---|---|
| `FONTMAKER_DEVICE` | `auto` | `cuda`／`mps`／`cpu`，auto 會自動挑最快的 |
| `FONTMAKER_BATCH` | 依裝置 | 一次生成幾個字（CUDA 32、MPS 16、CPU 8）；記憶體不足時調小 |
| `FONTMAKER_PORT` | `8765` | 網頁埠號 |
| `FONTMAKER_HOST` | `127.0.0.1` | 改成 `0.0.0.0` 可讓區網其他電腦連線 |
| `FONTMAKER_WORK` | `work/` | 專案、生成結果與字型掃描快取存放處 |
| `FONTMAKER_FONTS` | `fonts/` | 自己放的字型資料夾（優先於系統字型） |
| `FONTMAKER_EXTRA_FONT_DIRS` | — | 其他要掃描的字型資料夾，用 `:`（Windows 用 `;`）分隔 |
| `FONTDIFFUSER_DIR` | `vendor/FontDiffuser` | FontDiffuser 原始碼位置 |
| `FONTDIFFUSER_CKPT` | `vendor/ckpt` | 權重位置 |

## 程式結構

```
fontmaker/
  segment.py     圖片 → 二值化 → 分行 → 切字，做出風格參考圖
  analyze.py     文字特色分析（粗細、對比、傾斜、字寬、圓潤度、手寫感、字重）
  fontscan.py    掃描 fonts/ 與系統字型（中文涵蓋率、英數字特徵），結果快取
  content.py     內容字型（大小正規化、多字型備援），畫出每個字的骨架圖並記錄位置
  generator.py   FontDiffuser 推論（自動選 CUDA／MPS／CPU，批次生成）
  quality.py     多參考字挑最佳、筆畫粗細一致化
  latin.py       英數字：挑最接近的字型，套傾斜與粗細
  vectorize.py   點陣 → 去雜點 → 粗細調整 → potrace 曲線 → TrueType 輪廓
  fontbuild.py   組成 .ttf（名稱強制加 NC、字重、字距）
  charsets.py    字集（Big5 常用／次常用、英數、標點）
  jobs.py        專案管理與背景生成佇列（可暫停、續跑）
  server.py      網頁 API
web/             網頁介面
scripts/setup_fontdiffuser.py  下載 FontDiffuser、檢查權重與內容字型
```

## 已知限制

- 圖片目前只支援**橫式排列**；字要清楚、背景乾淨，每個字最好 60px 以上。
- FontDiffuser 輸出 96×96 像素，極細的筆畫細節會流失。
- 文字特色分析與英數字比對是依影像特徵估計，結果僅供參考，網頁上都可以手動改選。

## 開發與測試

```bash
pip install -r requirements-dev.txt
FONTMAKER_TEST_FONT=/path/to/任一含中文的字型.ttf pytest
```

測試用假的生成器取代 AI 模型，涵蓋切字、特徵分析、圓潤度判斷、向量化後的字形位置與大小、字型名稱 NC 標記、多參考字、英數字字型比對、API 全流程，以及暫停後重啟續跑。
