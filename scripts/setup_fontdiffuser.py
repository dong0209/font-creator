"""下載 FontDiffuser 原始碼到 vendor/FontDiffuser，並檢查權重檔是否就位。

FontDiffuser 的程式碼與權重不放進本專案，由這個腳本下載原始碼、使用者自行下載權重。

用法：python scripts/setup_fontdiffuser.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fontmaker import config  # noqa: E402

REPO = "https://github.com/yeungchenwa/FontDiffuser.git"
# 本專案測試過的版本
COMMIT = "7b28ce9c3b357f4fb23296622f458cf169803539"
CKPT_URL = "https://drive.google.com/drive/folders/12hfuZ9MQvXqcteNuz7JQ2B_mUcTr-5jZ"


def main() -> int:
    target = config.FONTDIFFUSER_DIR
    if (target / "src" / "model.py").is_file():
        print(f"✓ FontDiffuser 原始碼已存在：{target}")
    else:
        if not shutil.which("git"):
            print("✗ 找不到 git，請先安裝 git（macOS：xcode-select --install）")
            return 1
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"下載 FontDiffuser 到 {target} …")
        subprocess.run(["git", "clone", REPO, str(target)], check=True)
        subprocess.run(["git", "-C", str(target), "checkout", "-q", COMMIT], check=True)
        print("✓ FontDiffuser 原始碼下載完成")

    config.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    missing = config.missing_ckpt_files()
    if missing:
        print()
        print("✗ 還缺權重檔：" + "、".join(missing))
        print(f"  1. 打開 {CKPT_URL}")
        print("  2. 下載 ckpt 資料夾中的 unet.pth、content_encoder.pth、style_encoder.pth")
        print(f"  3. 放到 {config.CKPT_DIR}")
    else:
        print(f"✓ 權重檔已就位：{config.CKPT_DIR}")

    config.FONTS_DIR.mkdir(parents=True, exist_ok=True)
    fonts = config.list_content_fonts()
    if any("kaixin" in f.name.lower() for f in fonts):
        print("✓ 已有開心宋體 A（FontDiffuser 訓練用的內容字型）")
    else:
        print()
        print("△ 建議下載「開心宋體 A」（KaiXinSongA.ttf）放到", config.FONTS_DIR)
        print("  這是 FontDiffuser 訓練時用的內容字型，效果最好；沒有的話會改用電腦裡的宋體／明體。")
    return 0 if not missing else 2


if __name__ == "__main__":
    sys.exit(main())
