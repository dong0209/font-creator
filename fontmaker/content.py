"""內容字型：把每個字畫成 FontDiffuser 要的內容圖，並記下字在字身框中的位置。

FontDiffuser 訓練時的內容圖做法（utils.ttf2im）：字級 128px、以字的墨跡外框置中於 128×128 白底，
超出才縮小。這裡用 Pillow 重現同樣做法，並記錄墨跡外框在 em 座標中的位置，
之後把生成圖貼回字型時，才能還原基線、大小與上下位置（例如 g、y 的下伸部）。
"""

from dataclasses import dataclass
from pathlib import Path

from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

CANVAS = 128


@dataclass
class Placement:
    """內容字的位置資訊，單位皆為 em（1.0 = 一個字身寬），y 軸向上、基線為 0。"""

    cx: float  # 墨跡外框中心
    cy: float
    scale: float  # 畫進 128px 畫布時的縮放（通常為 1，太大的字才 < 1）
    advance: float  # 內容字型的字寬
    lsb: float  # 左側間距
    rsb: float  # 右側間距


class ContentFont:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        ttc_index = 0
        self._tt = TTFont(str(self.path), fontNumber=ttc_index, lazy=True)
        self.cmap = self._tt.getBestCmap() or {}
        self.upm = self._tt["head"].unitsPerEm
        self._pil = ImageFont.truetype(str(self.path), CANVAS, index=ttc_index)

    @property
    def name(self) -> str:
        return self.path.name

    def has(self, ch: str) -> bool:
        return ord(ch) in self.cmap

    def render(self, ch: str) -> tuple[Image.Image, Placement] | None:
        """回傳 128×128 RGB 內容圖與位置資訊；字型沒有這個字或沒有墨跡時回傳 None。"""
        if not self.has(ch):
            return None
        left, top, right, bottom = self._pil.getbbox(ch, anchor="ls")
        w, h = right - left, bottom - top
        if w <= 0 or h <= 0:
            return None
        advance = self._pil.getlength(ch)

        glyph = Image.new("L", (w, h), 0)
        ImageDraw.Draw(glyph).text((-left, -top), ch, font=self._pil, fill=255, anchor="ls")

        scale = 1.0
        if w > CANVAS or h > CANVAS:
            scale = CANVAS / max(w, h)
            glyph = glyph.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)

        canvas = Image.new("L", (CANVAS, CANVAS), 255)
        gw, gh = glyph.size
        x, y = round((CANVAS - gw) / 2), round((CANVAS - gh) / 2)
        canvas.paste(Image.new("L", glyph.size, 0), (x, y), glyph)

        placement = Placement(
            cx=(left + right) / 2 / CANVAS,
            cy=-(top + bottom) / 2 / CANVAS,
            scale=scale,
            advance=advance / CANVAS,
            lsb=left / CANVAS,
            rsb=(advance - right) / CANVAS,
        )
        return canvas.convert("RGB"), placement

    def space_advance(self) -> float:
        return self._pil.getlength(" ") / CANVAS if self.has(" ") else 0.25
