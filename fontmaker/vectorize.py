"""點陣字圖 → TrueType 輪廓。

流程：放大 4 倍 → 二值化 → 去雜點、補小洞 →（可選）以距離場調整筆畫粗細 → 輕微平滑 →
potrace 擬合三次貝茲曲線 → cu2qu 轉成 TrueType 二次曲線。
"""

import cv2
import numpy as np
import potrace
from fontTools.pens.cu2quPen import Cu2QuPen

from .content import CANVAS, Placement

UPSCALE = 4


# ---------- 點陣處理 ----------

def ink_mask(gray: np.ndarray, upscale: int = UPSCALE) -> np.ndarray:
    """灰階（白底黑字）→ 放大後的墨跡 float 遮罩（0–1，邊緣保留灰階以便平滑）。"""
    g = gray.astype(np.float32) / 255.0
    if upscale != 1:
        g = cv2.resize(g, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    return np.clip(1.0 - g, 0, 1)


def clean(ink: np.ndarray, min_area: float, max_hole: float) -> np.ndarray:
    """去掉面積小於 min_area 的墨點、填平面積小於 max_hole 的小洞（生成圖常見的瑕疵）。"""
    binary = (ink >= 0.5).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    small = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] < min_area]
    if small:
        binary[np.isin(labels, small)] = 0
    holes = (1 - binary).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=4)
    h, w = binary.shape
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        touches_border = x == 0 or y == 0 or x + bw == w or y + bh == h
        if not touches_border and area < max_hole:
            binary[labels == i] = 1
    return binary.astype(bool)


def stroke_width(binary: np.ndarray) -> float:
    """平均筆畫寬度（像素）≈ 2 × 面積 ÷ 周長。"""
    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    return 2 * float(binary.sum()) / max(perimeter, 1.0)


def adjust_weight(binary: np.ndarray, delta: float) -> np.ndarray:
    """以距離場把筆畫外擴（delta > 0）或內縮（delta < 0）delta 像素。"""
    if abs(delta) < 0.25:
        return binary
    pad = int(abs(delta)) + 2
    b = np.pad(binary.astype(np.uint8), pad)
    inside = cv2.distanceTransform(b, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - b, cv2.DIST_L2, 5)
    sdf = outside - inside  # 墨跡內為負
    out = sdf < delta
    return out[pad:-pad, pad:-pad]


def smooth(binary: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    blurred = cv2.GaussianBlur(binary.astype(np.float32), (0, 0), sigma)
    return blurred >= 0.5


# ---------- 描圖 ----------

def trace_bitmap(binary: np.ndarray, to_font, turdsize: int = 16) -> list[list[tuple]]:
    """回傳字型座標的輪廓，每個輪廓為 [("M", p), ("L", p) | ("C", c1, c2, p), ...]。

    外框順時針、內洞逆時針（TrueType 慣例）。
    """
    if not binary.any():
        return []
    path = potrace.Bitmap(~binary).trace(turdsize=turdsize, alphamax=1.0, opticurve=True, opttolerance=0.2)
    contours = []
    for curve in path.curves:
        start = to_font(curve.start_point.x, curve.start_point.y)
        ops = [("M", start)]
        for seg in curve.segments:
            if seg.is_corner:
                ops.append(("L", to_font(seg.c.x, seg.c.y)))
                ops.append(("L", to_font(seg.end_point.x, seg.end_point.y)))
            else:
                ops.append(("C", to_font(seg.c1.x, seg.c1.y), to_font(seg.c2.x, seg.c2.y),
                            to_font(seg.end_point.x, seg.end_point.y)))
        contours.append(ops)
    return _orient(contours)


def _polygon(ops) -> np.ndarray:
    pts = []
    for op in ops:
        if op[0] == "C":
            (x0, y0) = pts[-1]
            (x1, y1), (x2, y2), (x3, y3) = op[1], op[2], op[3]
            for t in (0.25, 0.5, 0.75, 1.0):
                mt = 1 - t
                pts.append((mt**3 * x0 + 3 * mt * mt * t * x1 + 3 * mt * t * t * x2 + t**3 * x3,
                            mt**3 * y0 + 3 * mt * mt * t * y1 + 3 * mt * t * t * y2 + t**3 * y3))
        else:
            pts.append(op[1])
    return np.array(pts, dtype=np.float64)


def _area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)) / 2


def _inside(pt, poly: np.ndarray) -> bool:
    x, y = pt
    xs, ys = poly[:, 0], poly[:, 1]
    xn, yn = np.roll(xs, -1), np.roll(ys, -1)
    cond = (ys > y) != (yn > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xcross = (xn - xs) * (y - ys) / (yn - ys) + xs
    return bool(np.count_nonzero(cond & (x < xcross)) % 2)


def _reverse(ops):
    pts = [ops[0][1]]
    segs = []
    for op in ops[1:]:
        segs.append((op, pts[-1]))
        pts.append(op[-1])
    out = [("M", pts[-1])]
    for op, prev in reversed(segs):
        if op[0] == "C":
            out.append(("C", op[2], op[1], prev))
        else:
            out.append(("L", prev))
    return out


def _orient(contours):
    polys = [_polygon(c) for c in contours]
    result = []
    for i, (ops, poly) in enumerate(zip(contours, polys)):
        depth = sum(1 for j, other in enumerate(polys) if j != i and _inside(poly[0], other))
        is_hole = depth % 2 == 1
        clockwise = _area(poly) < 0
        result.append(_reverse(ops) if clockwise == is_hole else ops)
    return result


def glyph_mapper(image_size: int, placement: Placement, upm: int, upscale: int = UPSCALE):
    """生成圖像素座標 → 字型座標（依內容字的位置還原基線與大小）。"""
    px_to_canvas = CANVAS / (image_size * upscale)
    k = upm / (CANVAS * placement.scale)

    def to_font(u: float, v: float) -> tuple[float, float]:
        cu, cv = u * px_to_canvas, v * px_to_canvas
        return (placement.cx * upm + (cu - CANVAS / 2) * k, placement.cy * upm - (cv - CANVAS / 2) * k)

    return to_font


def prepare(image, weight_delta: float = 0.0) -> np.ndarray:
    """生成圖（96px 灰階或 PIL）→ 清理、調整粗細、平滑後的 4 倍墨跡。weight_delta 以 96px 像素計。"""
    gray = np.array(image.convert("L")) if hasattr(image, "convert") else image
    size = gray.shape[1]
    ink = ink_mask(gray)
    scale = UPSCALE * size / 96
    binary = clean(ink, min_area=(3 * scale) ** 2, max_hole=(2 * scale) ** 2)
    binary = adjust_weight(binary, weight_delta * UPSCALE)
    return smooth(binary)


def trace(image, placement: Placement, upm: int, weight_delta: float = 0.0):
    gray = np.array(image.convert("L")) if hasattr(image, "convert") else image
    binary = prepare(gray, weight_delta)
    return trace_bitmap(binary, glyph_mapper(gray.shape[1], placement, upm))


# ---------- 輸出 ----------

def draw(contours, pen, dx: float = 0.0, dy: float = 0.0) -> None:
    """把輪廓畫到 TrueType pen（三次曲線自動轉二次）。"""
    qpen = Cu2QuPen(pen, max_err=1.0, reverse_direction=False)
    shift = lambda p: (p[0] + dx, p[1] + dy)  # noqa: E731
    for ops in contours:
        qpen.moveTo(shift(ops[0][1]))
        for op in ops[1:]:
            if op[0] == "L":
                qpen.lineTo(shift(op[1]))
            else:
                qpen.curveTo(shift(op[1]), shift(op[2]), shift(op[3]))
        qpen.closePath()


def bounds(contours) -> tuple[float, float] | None:
    xs = [p[0] for ops in contours for op in ops for p in op[1:]]
    return (min(xs), max(xs)) if xs else None
