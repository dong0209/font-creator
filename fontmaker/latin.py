"""英數字：FontDiffuser 以中文訓練，英數字品質差，所以改從現成字型中挑出風格最接近的一套，
再套上圖中分析出的傾斜與筆畫粗細，描成輪廓放進輸出字型。
"""

import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import analyze, vectorize
from .fontscan import FaceInfo

SAMPLE = "HOnoahgeABlbdpqmuvx"
SLANT_SAMPLE = "lIHdb"
FEATURE_SIZE = 120
RENDER_SIZE = 600
# 同一套字型中，英文筆畫（以字身為單位）約為中文的 1.32 倍（以 Noto Sans TC／DejaVu Sans 校正）
LATIN_TO_CJK_STROKE = 1.32
HANDWRITING = re.compile(r"hand|script|brush|marker|chalk|comic|felt|pen|crayon|手寫|手書|書法|楷|行書", re.I)
SYMBOLIC = re.compile(r"symbol|dingbat|wingding|webding|emoji|icon|math|braille|ornament|unifont|pixel|bitmap|LastResort", re.I)
LATIN_CHARS = "".join(chr(c) for c in range(0x21, 0x7F))


def _render(font: ImageFont.FreeTypeFont, ch: str) -> np.ndarray | None:
    size = int(font.size)
    im = Image.new("L", (3 * size, 3 * size), 0)
    ImageDraw.Draw(im).text((size, 2 * size), ch, font=font, fill=255, anchor="ls")
    a = np.array(im) > 127
    ys, xs = np.nonzero(a)
    if len(xs) == 0:
        return None
    return a[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def font_features(face: FaceInfo) -> dict | None:
    """英數字的風格特徵：筆畫粗細（字身比例）、粗細對比、邊角圓潤度、傾斜。"""
    try:
        font = ImageFont.truetype(face.path, FEATURE_SIZE, index=face.index)
    except OSError:
        return None
    strokes, contrasts, rounds = [], [], []
    for ch in SAMPLE:
        ink = _render(font, ch)
        if ink is None or ink.sum() < 20:
            continue
        sw = analyze._stroke_width(ink)
        strokes.append(sw / FEATURE_SIZE)
        c = analyze._contrast(ink)
        if c is not None:
            contrasts.append(c)
        r = analyze._roundness(ink, sw)
        if r is not None:
            rounds.append(r)
    if len(strokes) < 5:
        return None
    line = Image.new("L", (FEATURE_SIZE * 6, FEATURE_SIZE * 2), 0)
    ImageDraw.Draw(line).text((20, 20), SLANT_SAMPLE, font=font, fill=255)
    slant = analyze._slant(np.array(line) > 127)
    return {
        "stroke": float(np.median(strokes)),
        "contrast": float(np.median(contrasts)) if contrasts else 1.0,
        "roundness": float(np.median(rounds)) if rounds else None,
        "slant": float(slant),
    }


def _feature(analysis: dict, key: str):
    for f in analysis.get("features", []):
        if f["key"] == key:
            return f["value"]
    return None


def target_from_analysis(analysis: dict) -> dict:
    stroke = (_feature(analysis, "stroke") or 7.0) / 100
    roundness = _feature(analysis, "roundness")
    return {
        "stroke": stroke * LATIN_TO_CJK_STROKE,
        "contrast": _feature(analysis, "contrast") or 1.3,
        "roundness": None if roundness is None else roundness / 100,
        "slant": analysis.get("slant", 0.0),
        "handwriting": analysis.get("style", "").startswith("手寫"),
    }


def distance(target: dict, face: FaceInfo) -> float:
    f = face.features
    if not f:
        return 99.0
    d = 1.0 * abs(np.log(f["stroke"] / target["stroke"]))
    d += 1.5 * abs(np.log(f["contrast"] / target["contrast"]))
    if f["roundness"] is not None and target["roundness"] is not None:
        d += 1.0 * abs(f["roundness"] - target["roundness"])
    d += 0.3 * abs(f["slant"] - target["slant"]) / 15
    name = face.family + " " + face.path
    if target["handwriting"] != bool(HANDWRITING.search(name)):
        d += 0.5
    if SYMBOLIC.search(name):
        d += 10
    return float(d)


def rank(target: dict, faces: list[FaceInfo], limit: int = 8) -> list[tuple[FaceInfo, float]]:
    scored = sorted(((f, distance(target, f)) for f in faces if f.features), key=lambda x: x[1])
    # 同一家族只留最接近的一個
    out, seen = [], set()
    for f, d in scored:
        if f.family in seen:
            continue
        seen.add(f.family)
        out.append((f, d))
        if len(out) >= limit:
            break
    return out


class LatinStyler:
    """把選定字型的英數字套上目標傾斜與粗細後描成輪廓。"""

    def __init__(self, face: FaceInfo, target: dict, upm: int = 1000):
        self.face = face
        self.upm = upm
        self.font = ImageFont.truetype(face.path, RENDER_SIZE, index=face.index)
        feats = face.features or font_features(face) or {"stroke": target["stroke"], "slant": 0.0}
        self.shear = 0.0
        slant_diff = target["slant"] - feats["slant"]
        if abs(slant_diff) > 2:
            self.shear = float(np.tan(np.radians(slant_diff)))
        # 粗細修正（兩側各一半），最多 ±35%
        change = np.clip(target["stroke"] - feats["stroke"], -0.35 * feats["stroke"], 0.35 * feats["stroke"])
        self.delta_px = float(change * RENDER_SIZE / 2) if abs(change) / feats["stroke"] > 0.08 else 0.0

    def space_advance(self) -> int:
        return round(self.font.getlength(" ") * self.upm / RENDER_SIZE)

    def glyph(self, ch: str):
        """回傳 (輪廓, 字寬)；字型沒有這個字時回傳 None。"""
        if not self.font.getmask(ch).getbbox():
            return None
        size = RENDER_SIZE
        pad = int(abs(self.delta_px)) + 4
        shear_room = int(abs(self.shear) * 1.5 * size) + pad
        w, h = 3 * size + 2 * shear_room, 3 * size
        ox, oy = size + shear_room, 2 * size  # 基線原點在畫布中的位置
        im = Image.new("L", (w, h), 0)
        ImageDraw.Draw(im).text((ox, oy), ch, font=self.font, fill=255, anchor="ls")
        ink = np.array(im, dtype=np.float32) / 255
        if self.shear:
            # 以基線為軸水平錯切：越高的點往右（正傾斜）移越多
            m = np.float32([[1, -self.shear, self.shear * oy], [0, 1, 0]])
            ink = cv2.warpAffine(ink, m, (w, h), flags=cv2.INTER_LINEAR)
        binary = ink >= 0.5
        ys, xs = np.nonzero(binary)
        if len(xs) == 0:
            return None
        # 裁到墨跡外框（留邊給外擴與平滑），後續處理只在小範圍內做
        m = pad + 8
        x0, y0 = max(xs.min() - m, 0), max(ys.min() - m, 0)
        x1, y1 = min(xs.max() + m + 1, w), min(ys.max() + m + 1, h)
        binary = binary[y0:y1, x0:x1]
        ox, oy = ox - x0, oy - y0
        binary = vectorize.adjust_weight(binary, self.delta_px)
        binary = vectorize.smooth(binary, sigma=1.5)
        k = self.upm / RENDER_SIZE
        dx = self.delta_px

        def to_font(u: float, v: float) -> tuple[float, float]:
            return ((u - ox + dx) * k, (oy - v) * k)

        contours = vectorize.trace_bitmap(binary, to_font, turdsize=int((RENDER_SIZE / 100) ** 2))
        advance = round((self.font.getlength(ch) + 2 * self.delta_px) * k)
        return contours, max(advance, 1)
