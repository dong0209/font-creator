"""把描好的輪廓組成 TrueType 字型檔。

非商業用途標記：字型家族名稱一律帶「NC」字尾（例如「MyHand NC」），程式不提供移除選項。
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen

from . import vectorize
from .content import Placement

UPM = 1000
ASCENT, DESCENT = 880, -120
NC_SUFFIX = "NC"


def nc_family_name(name: str) -> str:
    """確保字型名稱以「 NC」結尾。"""
    name = re.sub(r"\s+", " ", name).strip() or "Generated"
    if re.search(rf"(^|\s){NC_SUFFIX}$", name, flags=re.IGNORECASE):
        name = name[: -len(NC_SUFFIX)].rstrip()
    return f"{name} {NC_SUFFIX}".strip()


def _ps_name(family: str) -> str:
    ascii_only = re.sub(r"[^A-Za-z0-9-]", "", family.replace(" ", "-"))
    return (ascii_only.strip("-") or "GeneratedNC")[:63]


def _is_fullwidth(ch: str) -> bool:
    cp = ord(ch)
    return cp >= 0x2E80 or 0xFF01 <= cp <= 0xFF60


@dataclass
class GlyphSource:
    char: str
    image: object = None  # PIL.Image 或灰階 numpy 陣列（AI 生成結果）
    placement: Placement | None = None
    weight_delta: float = 0.0  # 筆畫粗細修正（96px 像素）
    contours: list | None = None  # 已描好的輪廓（英數字）
    advance: int | None = None  # 搭配 contours 使用的字寬


@dataclass
class FontSpec:
    family: str
    weight: int = 400
    italic_angle: float = 0.0
    space_advance: float = 0.25
    glyphs: list[GlyphSource] = field(default_factory=list)


def _notdef():
    pen = TTGlyphPen(None)
    for pts in ([(100, 0), (100, 800), (900, 800), (900, 0)], [(150, 50), (850, 50), (850, 750), (150, 750)]):
        pen.moveTo(pts[0])
        for p in pts[1:]:
            pen.lineTo(p)
        pen.closePath()
    return pen.glyph()


def build_font(spec: FontSpec, out_path: str | Path) -> dict:
    family = nc_family_name(spec.family)
    order = [".notdef", "space", "uni3000"]
    cmap = {0x20: "space", 0x3000: "uni3000"}
    glyf = {".notdef": _notdef(), "space": TTGlyphPen(None).glyph(), "uni3000": TTGlyphPen(None).glyph()}
    metrics = {".notdef": (UPM, 100), "space": (round(spec.space_advance * UPM), 0), "uni3000": (UPM, 0)}
    empty = []

    for src in spec.glyphs:
        cp = ord(src.char)
        if cp in cmap:
            continue
        name = f"uni{cp:04X}" if cp <= 0xFFFF else f"u{cp:05X}"
        if src.contours is not None:
            contours = src.contours
        else:
            contours = vectorize.trace(src.image, src.placement, UPM, src.weight_delta)
        span = vectorize.bounds(contours)
        if span is None:
            empty.append(src.char)
            continue
        p = src.placement
        dy = 0.0
        if src.advance is not None:
            dx, advance = 0.0, src.advance
        elif _is_fullwidth(src.char):
            # 全形字：把內容字型的字身框中心對齊輸出字型的字身框中心
            dx = (0.5 - p.advance / 2) * UPM
            dy = (ASCENT + DESCENT) / 2 - p.em_cy * UPM
            advance = UPM
        else:
            # 半形字：依生成後的實際字寬重算字距，左右留白沿用內容字型
            dx = p.lsb * UPM - span[0]
            advance = round(p.lsb * UPM + (span[1] - span[0]) + p.rsb * UPM)
        pen = TTGlyphPen(None)
        vectorize.draw(contours, pen, dx, dy)
        glyf[name] = pen.glyph(dropImpliedOnCurves=True)
        order.append(name)
        cmap[cp] = name
        metrics[name] = (max(advance, 1), 0)

    fb = FontBuilder(UPM, isTTF=True)
    fb.setupGlyphOrder(order)
    fb.setupCharacterMap(cmap)
    fb.setupGlyf(glyf)
    glyph_table = fb.font["glyf"]
    hmtx = {}
    for name in order:
        g = glyph_table[name]
        g.recalcBounds(glyph_table)
        hmtx[name] = (metrics[name][0], getattr(g, "xMin", 0))
    fb.setupHorizontalMetrics(hmtx)
    fb.setupHorizontalHeader(ascent=ASCENT, descent=DESCENT)
    fb.setupNameTable({
        "familyName": family,
        "styleName": "Regular",
        "uniqueFontIdentifier": f"{_ps_name(family)}-Regular",
        "fullName": family,
        "psName": f"{_ps_name(family)}-Regular",
        "version": "Version 1.000",
    })
    fb.setupOS2(
        usWeightClass=spec.weight,
        sTypoAscender=ASCENT, sTypoDescender=DESCENT, sTypoLineGap=0,
        usWinAscent=ASCENT, usWinDescent=-DESCENT,
        ulCodePageRange1=(1 << 0) | (1 << 20),  # Latin 1、繁體中文 Big5
    )
    fb.setupPost(italicAngle=-abs(spec.italic_angle) if spec.italic_angle > 3 else 0)
    fb.setupHead(unitsPerEm=UPM)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fb.save(str(out_path))
    return {"family": family, "glyphs": len(order) - 3, "empty": empty, "path": str(out_path)}
