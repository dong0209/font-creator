"""把 FontDiffuser 生成的 96px 字圖描成 TrueType 二次曲線輪廓。

做法：放大 4 倍平滑 → 二值化 → 找輪廓 → 簡化頂點。銳利轉角的頂點設為線上點，
其餘當作控制點，TrueType 會在相鄰控制點中間自動補上線上點，得到平滑曲線。
"""

import math

import cv2
import numpy as np
from fontTools.pens.ttGlyphPen import TTGlyphPen

from .content import CANVAS, Placement

UPSCALE = 4
CORNER_DEG = 50
EPSILON = 1.2  # 簡化容許誤差（放大後像素）


def _clean(image: np.ndarray) -> np.ndarray:
    """回傳放大後的墨跡（uint8 0/1），並去掉生成時常見的小雜點。"""
    big = cv2.resize(image, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
    ink = (big < 128).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    min_area = (UPSCALE * 2) ** 2
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            ink[labels == i] = 0
    return ink


def _signed_area(pts: list[tuple[float, float]]) -> float:
    return sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1])) / 2


def _on_curve_flags(pts: np.ndarray) -> list[bool]:
    n = len(pts)
    flags = []
    for i in range(n):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
        v1, v2 = b - a, c - b
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            flags.append(True)
            continue
        cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1))
        flags.append(math.degrees(math.acos(cos)) > CORNER_DEG)
    if not any(flags):
        flags[0] = True
    return flags


def trace(image, placement: Placement, upm: int) -> list[list[tuple[float, float, bool]]]:
    """回傳字型座標的輪廓清單，每個點為 (x, y, 是否為線上點)。外框順時針、內洞逆時針。"""
    gray = np.array(image.convert("L")) if hasattr(image, "convert") else image
    ink = _clean(gray)
    contours, hierarchy = cv2.findContours(ink, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return []

    size = gray.shape[1]
    px_to_canvas = CANVAS / (size * UPSCALE)
    k = upm / (CANVAS * placement.scale)

    def to_font(u: float, v: float) -> tuple[float, float]:
        cu, cv = u * px_to_canvas, v * px_to_canvas
        return (placement.cx * upm + (cu - CANVAS / 2) * k, placement.cy * upm - (cv - CANVAS / 2) * k)

    result = []
    for contour, (_, _, _, parent) in zip(contours, hierarchy[0]):
        approx = cv2.approxPolyDP(contour, EPSILON, True).reshape(-1, 2).astype(float)
        if len(approx) < 3:
            continue
        pts = [to_font(u + 0.5, v + 0.5) for u, v in approx]
        is_hole = parent != -1
        # 字型座標 y 向上：順時針面積為負
        if (_signed_area(pts) < 0) == is_hole:
            pts.reverse()
            approx = approx[::-1]
        flags = _on_curve_flags(np.array(pts))
        result.append([(x, y, on) for (x, y), on in zip(pts, flags)])
    return result


def draw(contours, pen: TTGlyphPen, dx: float = 0.0) -> None:
    for contour in contours:
        start = next(i for i, p in enumerate(contour) if p[2])
        pts = contour[start:] + contour[:start]
        pen.moveTo((round(pts[0][0] + dx), round(pts[0][1])))
        offs = []
        for x, y, on in pts[1:] + pts[:1]:
            p = (round(x + dx), round(y))
            if on:
                if offs:
                    pen.qCurveTo(*offs, p)
                    offs = []
                else:
                    pen.lineTo(p)
            else:
                offs.append(p)
        pen.closePath()


def bounds(contours) -> tuple[float, float] | None:
    xs = [p[0] for c in contours for p in c]
    return (min(xs), max(xs)) if xs else None
