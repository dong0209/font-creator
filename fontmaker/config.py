"""路徑與執行設定，全部可用環境變數覆寫。"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _path(env: str, default: Path) -> Path:
    value = os.environ.get(env)
    return Path(value).expanduser().resolve() if value else default


# FontDiffuser 原始碼（由 scripts/setup_fontdiffuser.py 下載，不隨本專案散布）
FONTDIFFUSER_DIR = _path("FONTDIFFUSER_DIR", ROOT / "vendor" / "FontDiffuser")
# FontDiffuser 權重目錄，需含 unet.pth、content_encoder.pth、style_encoder.pth
CKPT_DIR = _path("FONTDIFFUSER_CKPT", ROOT / "vendor" / "ckpt")
# 內容字型：提供每個字的「骨架」，預設放在 fonts/ 底下
FONTS_DIR = _path("FONTMAKER_FONTS", ROOT / "fonts")
# 工作資料（上傳圖片、生成中的字圖、輸出字型）
WORK_DIR = _path("FONTMAKER_WORK", ROOT / "work")

# auto / cuda / mps / cpu
DEVICE = os.environ.get("FONTMAKER_DEVICE", "auto")
BATCH_SIZE = int(os.environ.get("FONTMAKER_BATCH", "0"))  # 0 = 依裝置自動決定

CKPT_FILES = ("unet.pth", "content_encoder.pth", "style_encoder.pth")

FONT_EXTS = (".ttf", ".otf", ".ttc")


def list_content_fonts() -> list[Path]:
    if not FONTS_DIR.is_dir():
        return []
    return sorted(p for p in FONTS_DIR.iterdir() if p.suffix.lower() in FONT_EXTS)


def missing_ckpt_files() -> list[str]:
    return [name for name in CKPT_FILES if not (CKPT_DIR / name).is_file()]
