"""下載 FontDiffuser 原始碼到 vendor/FontDiffuser，並檢查權重檔是否就位。

FontDiffuser 的 repo 沒有附授權條款，所以本專案不重新散布它的程式碼與權重，
由使用者自行下載，並自行評估使用上的授權風險。

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

    fonts = config.list_content_fonts()
    if fonts:
        print(f"✓ 內容字型：{', '.join(f.name for f in fonts)}")
    else:
        config.FONTS_DIR.mkdir(parents=True, exist_ok=True)
        print()
        print(f"✗ 還沒有內容字型，請把字型檔放到 {config.FONTS_DIR}（建議全字庫正宋體 TW-Sung，見 README）")
    return 0 if not missing and fonts else 2


if __name__ == "__main__":
    sys.exit(main())
