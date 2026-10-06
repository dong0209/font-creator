import os
import shutil
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

CANDIDATES = [
    os.environ.get("FONTMAKER_TEST_FONT", ""),
    "fonts/TW-Sung-98_1.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/msjh.ttc",
]


@pytest.fixture(scope="session")
def cjk_font_path() -> Path:
    for c in CANDIDATES:
        if c and Path(c).is_file():
            return Path(c).resolve()
    pytest.skip("找不到含中文的測試字型，請設定 FONTMAKER_TEST_FONT")


@pytest.fixture
def fonts_dir(tmp_path, cjk_font_path, monkeypatch):
    d = tmp_path / "fonts"
    d.mkdir()
    shutil.copy(cjk_font_path, d / cjk_font_path.name)
    from fontmaker import config
    monkeypatch.setattr(config, "FONTS_DIR", d)
    return d


def draw_text(font_path: Path, lines: list[str], size: int = 80, shear: float = 0.0,
              invert: bool = False) -> Image.Image:
    font = ImageFont.truetype(str(font_path), size)
    width = int(size * max(len(t) for t in lines) * 1.3) + 80
    height = int(size * 1.6 * len(lines)) + 80
    img = Image.new("L", (width, height), 255)
    d = ImageDraw.Draw(img)
    for i, t in enumerate(lines):
        d.text((40, 40 + i * size * 1.6), t, font=font, fill=0)
    if shear:
        img = img.transform(img.size, Image.AFFINE, (1, shear, -shear * height / 2, 0, 1, 0), fillcolor=255)
    if invert:
        img = Image.eval(img, lambda v: 255 - v)
    return img.convert("RGB")


class FakeGenerator:
    """以內容字圖直接當生成結果，用來測試 AI 以外的整條流程。"""

    batch_size = 4

    def __init__(self, steps: int = 20):
        self.steps = steps
        self.calls = 0

    def generate(self, contents, style, seed=123):
        self.calls += 1
        return [c.convert("RGB").resize((96, 96), Image.BILINEAR) for c in contents]
