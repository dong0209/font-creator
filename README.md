# font-creator：NC 字型產生器

上傳一張有文字的圖片，程式會：

1. **切出圖中的每個字**，並**分析文字特色**：筆畫粗細、粗細對比、傾斜角度、字寬比、邊角圓潤度、手寫感，推測字體類型（黑體／明體／圓體／手寫）與字重。
2. 讓你挑一個字當**風格參考**，用 AI 模型 [FontDiffuser](https://github.com/yeungchenwa/FontDiffuser) 以這個風格**生成整套中英文字**。
3. 把生成的字描成向量輪廓，輸出 **TrueType 字型檔（.ttf）**。

> **此程式產生的字型禁止用於商業用途。** 字型名稱一律自動加上「NC」（Non-Commercial）標記，例如輸入「MyHand」會得到「MyHand NC」。

## 先看這裡：限制與授權

| 項目 | 說明 |
|---|---|
| AI 模型授權 | FontDiffuser 的 repo **沒有附授權條款**。本專案不重新散布它的程式碼與權重，由安裝腳本從原作者處下載，**使用風險請自行評估**。 |
| 內容字型 | AI 需要一套「內容字型」提供每個字的骨架。請用**全字庫正宋體（TW-Sung）**：政府資料開放授權，允許衍生後另訂授權。**不要用 Noto／思源等 OFL 字型**：OFL 要求衍生字型也得用 OFL 發布，而 OFL 允許商用，和「禁止商業用途」互相衝突。 |
| 圖中字型的著作權 | 如果圖片裡是某套商業字型，模仿它做出的字型仍可能有爭議，請只用於個人、非商業用途。 |
| 生成速度 | 見下方「要跑多久」。 |
| 品質 | FontDiffuser 以中文字訓練，輸出 96×96 像素，所以英文字母品質較差，細節（如極細的襯線）會流失。 |
| 圖片要求 | 目前只支援**橫式排列**；字要清楚、背景乾淨，每個字最好 60px 以上。 |

## 要跑多久

FontDiffuser 是擴散模型，每個字要跑 20 次（可調）神經網路推論。

| 環境 | 每字（20 步） | 常用字 5,401 字 |
|---|---|---|
| 4 核心雲端 CPU（實測） | 約 28 秒 | 約 42 小時 |
| Apple M4（MPS GPU，未實測） | 估計 2–5 秒 | 估計 3–8 小時 |
| NVIDIA 獨立顯示卡 | 估計 < 1 秒 | 估計 1–2 小時 |

- 網頁上的「試生成」會量出你電腦的實際速度，並換算整套字型需要多久。
- 步數改成 10 可以快一倍，品質會稍差。
- 生成可以隨時**暫停**，關掉程式也沒關係：已生成的字會保留，下次按「繼續」接著跑。
- 預設字集是**常用字 5,401 字**（Big5 常用字區，依教育部常用國字表編排），另有「常用＋次常用 13,053 字」與「自訂字表」可選。如果要剛好用教育部 4,808 字的版本，把字表貼到「自訂字表」即可。

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

### 還需要手動放兩樣東西

`setup_fontdiffuser.py` 會下載 FontDiffuser 原始碼，並告訴你還缺什麼：

1. **模型權重**：從 [FontDiffuser 的 Google Drive](https://drive.google.com/drive/folders/12hfuZ9MQvXqcteNuz7JQ2B_mUcTr-5jZ) 下載 `unet.pth`、`content_encoder.pth`、`style_encoder.pth`，放到 `vendor/ckpt/`。
2. **內容字型**：到[全字庫網站](https://www.cns11643.gov.tw/)下載「正宋體」（TW-Sung），把 `.ttf` 放到 `fonts/`。

再執行一次 `python scripts/setup_fontdiffuser.py`，全部打勾就完成了。

## 使用

```bash
python -m fontmaker.server
```

打開 <http://127.0.0.1:8765>：

1. **上傳圖片** → 自動切字並顯示「文字特色分析」。
2. **選風格參考字**（已依適合程度排序，筆畫多、方正、清楚的中文字最好）→ 按「試生成」看效果，可以多換幾個參考字或隨機種子比較。
3. 滿意後按「用這組設定生成完整字型」，輸入字型名稱、選字集 → **開始生成**。
4. 完成後**下載 .ttf**，也可以直接在頁面上試打。

## 設定（環境變數）

| 變數 | 預設 | 說明 |
|---|---|---|
| `FONTMAKER_DEVICE` | `auto` | `cuda`／`mps`／`cpu`，auto 會自動挑最快的 |
| `FONTMAKER_BATCH` | 依裝置 | 一次生成幾個字（CUDA 32、MPS 16、CPU 8）；記憶體不足時調小 |
| `FONTMAKER_PORT` | `8765` | 網頁埠號 |
| `FONTMAKER_HOST` | `127.0.0.1` | 改成 `0.0.0.0` 可讓區網其他電腦連線 |
| `FONTMAKER_WORK` | `work/` | 專案與生成結果存放處 |
| `FONTMAKER_FONTS` | `fonts/` | 內容字型資料夾 |
| `FONTDIFFUSER_DIR` | `vendor/FontDiffuser` | FontDiffuser 原始碼位置 |
| `FONTDIFFUSER_CKPT` | `vendor/ckpt` | 權重位置 |

## 程式結構

```
fontmaker/
  segment.py     圖片 → 二值化 → 分行 → 切字，做出風格參考圖
  analyze.py     文字特色分析（粗細、對比、傾斜、字寬、圓潤度、手寫感、字重）
  content.py     用內容字型畫出每個字的骨架圖，並記錄位置（基線、大小）
  generator.py   FontDiffuser 推論（自動選 CUDA／MPS／CPU，批次生成）
  vectorize.py   生成的字圖 → TrueType 二次曲線輪廓
  fontbuild.py   組成 .ttf（名稱強制加 NC、字重、字距）
  charsets.py    字集（Big5 常用／次常用、英數、標點）
  jobs.py        專案管理與背景生成佇列（可暫停、續跑）
  server.py      網頁 API
web/             網頁介面
scripts/setup_fontdiffuser.py  下載 FontDiffuser、檢查權重與內容字型
```

## 開發與測試

```bash
pip install -r requirements-dev.txt
FONTMAKER_TEST_FONT=/path/to/任一含中文的字型.ttf pytest
```

測試用假的生成器取代 AI 模型，涵蓋切字、特徵分析、向量化後的字形位置、字型名稱 NC 標記、API 全流程，以及暫停後重啟續跑。
