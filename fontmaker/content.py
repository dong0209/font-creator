"""內容字型：把每個字畫成 FontDiffuser 要的內容圖，並記下字在字身框中的位置。

對齊 FontDiffuser 的訓練條件（utils.ttf2im 與 data_examples）：
- 字以墨跡外框置中於 128×128 白底；
- 訓練用的內容字型（開心宋體 A）中，「國」這類滿版字約佔畫布 92%。
  不同字型天生大小不同，所以每套字型都先量「國」的大小，調整字級讓它也佔 92%，
  這樣換字型或用備援字型補字時，模型看到的內容圖大小一致。

座標單位：「正規化 em」= 畫布像素 ÷ 128。輸出字型時 1.0 對應 1000 units，
因此所有字（不論出自哪套內容字型）在輸出字型中大小一致。
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

CANVAS = 128
TARGET_FILL = 0.92
PROBE_CHARS = "國圖龍永"


@dataclass
class Placement:
    """內容字的位置資訊（正規化 em，y 軸向上、基線為 0）。"""

    cx: float  # 墨跡外框中心
    cy: float
    scale: float  # 畫進 128px 畫布時的額外縮放（通常為 1，太大的字才 < 1）
    advance: float
    lsb: float
    rsb: float
    em_cy: float = 0.38  # 內容字型字身框的垂直中心（全形字對齊用）


class ContentFont:
    def __init__(self, path: str | Path, index: int = 0):
        self.path = Path(path)
        self.index = index
        tt = TTFont(str(self.path), fontNumber=index, lazy=True)
        self.cmap = tt.getBestCmap() or {}
        self.size = self._normalized_size()
        upm = tt["head"].unitsPerEm
        os2 = tt.get("OS/2")
        if os2 is not None and os2.sTypoAscender - os2.sTypoDescender > 0:
            asc, desc = os2.sTypoAscender, os2.sTypoDescender
        else:
            asc, desc = tt["hhea"].ascent, tt["hhea"].descent
        # 字身框中心，換算成正規化 em
        self.em_cy = (asc + desc) / 2 / upm * self.size / CANVAS
        self._pil = ImageFont.truetype(str(self.path), self.size, index=index)

    @property
    def name(self) -> str:
        return self.path.name

    def _ink(self, font: ImageFont.FreeTypeFont, ch: str, size: int):
        """實際畫出來量墨跡外框（Pillow 的 getbbox 水平方向不貼齊墨跡）。

        回傳 (墨跡遮罩, left, top, right, bottom)，座標相對於基線原點、y 向下。
        """
        ox, oy = size, 2 * size
        im = Image.new("L", (3 * size, 3 * size), 0)
        ImageDraw.Draw(im).text((ox, oy), ch, font=font, fill=255, anchor="ls")
        a = np.asarray(im)
        ys, xs = np.nonzero(a > 8)
        if len(xs) == 0:
            return None
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        return im.crop((x0, y0, x1, y1)), x0 - ox, y0 - oy, x1 - ox, y1 - oy

    def _normalized_size(self) -> int:
        probe = next((c for c in PROBE_CHARS if ord(c) in self.cmap), None)
        if probe is None:
            return CANVAS
        font = ImageFont.truetype(str(self.path), CANVAS, index=self.index)
        ink = self._ink(font, probe, CANVAS)
        if ink is None:
            return CANVAS
        _, left, top, right, bottom = ink
        extent = max(right - left, bottom - top)
        return max(16, round(CANVAS * TARGET_FILL * CANVAS / extent))

    def has(self, ch: str) -> bool:
        return ord(ch) in self.cmap

    def render(self, ch: str) -> tuple[Image.Image, Placement] | None:
        """回傳 128×128 RGB 內容圖與位置資訊；字型沒有這個字或沒有墨跡時回傳 None。"""
        if not self.has(ch):
            return None
        ink = self._ink(self._pil, ch, self.size)
        if ink is None:
            return None
        glyph, left, top, right, bottom = ink
        w, h = glyph.size
        advance = self._pil.getlength(ch)

        scale = 1.0
        if w > CANVAS or h > CANVAS:
            scale = CANVAS / max(w, h)
            glyph = glyph.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)

        canvas = Image.new("L", (CANVAS, CANVAS), 255)
        gw, gh = glyph.size
        x, y = round((CANVAS - gw) / 2), round((CANVAS - gh) / 2)
        canvas.paste(Image.new("L", glyph.size, 0), (x, y), glyph)

        # 畫布中心對應的字型座標（置中時的取整誤差一併補償）
        cx = left + (CANVAS / 2 - x) / scale
        cy = top + (CANVAS / 2 - y) / scale
        placement = Placement(
            cx=cx / CANVAS,
            cy=-cy / CANVAS,
            scale=scale,
            advance=advance / CANVAS,
            lsb=left / CANVAS,
            rsb=(advance - right) / CANVAS,
            em_cy=self.em_cy,
        )
        return canvas.convert("RGB"), placement

    def space_advance(self) -> float:
        return self._pil.getlength(" ") / CANVAS if self.has(" ") else 0.25


class ContentChain:
    """主內容字型加上備援字型：缺字時依序改用下一套。"""

    def __init__(self, fonts: list[ContentFont]):
        if not fonts:
            raise ValueError("沒有可用的內容字型")
        self.fonts = fonts

    @classmethod
    def from_faces(cls, faces) -> "ContentChain":
        fonts = []
        for f in faces:
            try:
                fonts.append(ContentFont(f.path, f.index))
            except Exception:  # noqa: BLE001 - 讀不了的字型就跳過
                continue
        return cls(fonts)

    @property
    def name(self) -> str:
        return self.fonts[0].name

    def font_for(self, ch: str) -> ContentFont | None:
        return next((f for f in self.fonts if f.has(ch)), None)

    def has(self, ch: str) -> bool:
        return self.font_for(ch) is not None

    def render(self, ch: str):
        for f in self.fonts:
            r = f.render(ch)
            if r is not None:
                return r
        return None

    def space_advance(self) -> float:
        return self.fonts[0].space_advance()
