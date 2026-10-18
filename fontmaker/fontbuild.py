"""把描好的輪廓組成 TrueType 字型檔。

非商業用途標記：字型家族名稱一律帶「NC」字尾（例如「MyHand NC」），程式不提供移除選項。
"""

import hashlib
import os
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
    """PostScript 名稱只能用英數字。中文名稱轉不出來時，用名稱的雜湊值區分，避免不同字型撞名。"""
    ascii_only = re.sub(r"[^A-Za-z0-9-]", "", family.replace(" ", "-")).strip("-")
    meaningful = re.sub(rf"-?{NC_SUFFIX}$", "", ascii_only, flags=re.IGNORECASE)
    if not meaningful:
        digest = hashlib.sha1(family.encode("utf-8")).hexdigest()[:8]
        ascii_only = f"Font{digest}-{NC_SUFFIX}"
    return ascii_only[:63]


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


def _trace_job(args):
    image, placement, weight_delta = args
    return vectorize.trace(image, placement, UPM, weight_delta)


def trace_all(sources: list[GlyphSource], progress=None) -> list:
    """描出所有需要描圖的字。字多時用多個 CPU 核心平行處理（5,000 字單核約十分鐘）。"""
    todo = [i for i, src in enumerate(sources) if src.contours is None]
    results: dict[int, list] = {}
    workers = max(1, (os.cpu_count() or 2) - 1)
    jobs = [(sources[i].image, sources[i].placement, sources[i].weight_delta) for i in todo]
    if len(todo) >= 200 and workers > 1:
        try:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=workers) as pool:
                for n, contours in enumerate(pool.map(_trace_job, jobs, chunksize=32)):
                    results[todo[n]] = contours
                    if progress is not None and n % 100 == 0:
                        progress(n, len(todo))
        except Exception:  # noqa: BLE001 - 平行處理失敗（例如環境不允許開子程序）就改用單核
            results.clear()
    for n, i in enumerate(todo):
        if i not in results:
            results[i] = _trace_job(jobs[n])
            if progress is not None and n % 100 == 0:
                progress(n, len(todo))
    return [results.get(i, src.contours) for i, src in enumerate(sources)]


def build_font(spec: FontSpec, out_path: str | Path, progress=None) -> dict:
    """progress(已完成, 總數)：描圖很花時間，定期回報進度。"""
    family = nc_family_name(spec.family)
    order = [".notdef", "space", "uni3000"]
    cmap = {0x20: "space", 0x3000: "uni3000"}
    glyf = {".notdef": _notdef(), "space": TTGlyphPen(None).glyph(), "uni3000": TTGlyphPen(None).glyph()}
    metrics = {".notdef": (UPM, 100), "space": (round(spec.space_advance * UPM), 0), "uni3000": (UPM, 0)}
    empty = []

    traced = trace_all(spec.glyphs, progress)
    for src, contours in zip(spec.glyphs, traced):
        cp = ord(src.char)
        if cp in cmap:
            continue
        name = f"uni{cp:04X}" if cp <= 0xFFFF else f"u{cp:05X}"
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
