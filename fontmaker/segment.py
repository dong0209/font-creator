"""從圖片切出單字。

流程：灰階 → 二值化（自動判斷深字淺底或淺字深底）→ 連通元件 → 依垂直重疊分行 →
行內把上下疊（如「二」「i」）與左右相鄰的部件（如「川」「好」）合併成單字。
切字只用來取得風格參考圖與分析特徵，不需要辨識是哪個字。
"""

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageOps

MAX_SIDE = 3000
STYLE_SIZE = 96
# 方塊字高度約為字身的 86%（用來從行高推估字身大小）
FILL_RATIO = 0.86
# FontDiffuser 訓練資料中，風格字（目標字型）約佔畫布 80%，參考圖照這個比例放
STYLE_FILL = 0.80


@dataclass
class Box:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    def union(self, other: "Box") -> "Box":
        return Box(min(self.x0, other.x0), min(self.y0, other.y0), max(self.x1, other.x1), max(self.y1, other.y1))


@dataclass
class Glyph:
    index: int
    line: int
    box: Box
    line_height: float
    ink: np.ndarray  # bool，外框範圍內的墨跡
    score: float = 0.0  # 適合當風格參考的程度（0–1）


@dataclass
class Segmentation:
    ink: np.ndarray  # 整張圖的墨跡（bool）
    glyphs: list[Glyph] = field(default_factory=list)
    lines: list[list[int]] = field(default_factory=list)  # 每行包含的 glyph index
    inverted: bool = False  # 原圖是否為淺字深底


def load_gray(image: Image.Image) -> np.ndarray:
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(bg, rgba)
    gray = np.array(image.convert("L"))
    h, w = gray.shape
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        gray = cv2.resize(gray, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return gray


def binarize(gray: np.ndarray) -> tuple[np.ndarray, bool]:
    """回傳 (墨跡 bool 陣列, 是否反相)。"""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate([th[0], th[-1], th[:, 0], th[:, -1]])
    background_is_white = np.mean(border > 127) >= 0.5
    ink = th < 128 if background_is_white else th > 127
    if ink.mean() > 0.45:
        # 光線不均的照片：Otsu 會把大片陰影當成墨跡，改用區域門檻
        block = max(15, (min(gray.shape) // 20) | 1)
        th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 10)
        ink = th < 128 if background_is_white else th > 127
    return ink, not background_is_white


def _components(ink: np.ndarray) -> list[Box]:
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    if n <= 1:
        return []
    areas = stats[1:, cv2.CC_STAT_AREA]
    min_area = max(4, int(np.percentile(areas, 90) * 0.002))
    boxes = []
    for x, y, w, h, area in stats[1:]:
        if area < min_area:
            continue
        boxes.append(Box(int(x), int(y), int(x + w), int(y + h)))
    return boxes


def _vertical_overlap(a: Box, b: Box) -> float:
    inter = min(a.y1, b.y1) - max(a.y0, b.y0)
    return inter / max(1, min(a.h, b.h))


def _horizontal_overlap(a: Box, b: Box) -> float:
    inter = min(a.x1, b.x1) - max(a.x0, b.x0)
    return inter / max(1, min(a.w, b.w))


def _group_lines(boxes: list[Box]) -> list[list[Box]]:
    """依垂直重疊把部件分行。大部件先決定行，小部件（標點、i 的點）再依附上去。"""
    order = sorted(boxes, key=lambda b: -b.h)
    lines: list[tuple[Box, list[Box]]] = []
    for b in order:
        best, best_ov = None, 0.0
        for i, (span, _) in enumerate(lines):
            ov = _vertical_overlap(span, b)
            if ov > best_ov:
                best, best_ov = i, ov
        # 小部件只要和行有部分重疊或貼近即可併入
        small = lines and b.h < 0.5 * np.median([s.h for s, _ in lines])
        if best is not None and (best_ov >= 0.5 or (small and best_ov > 0)):
            span, members = lines[best]
            members.append(b)
            lines[best] = (span.union(b) if not small else span, members)
        elif small and lines:
            # 完全不重疊的小點（如 i、j 上的點），併到最近的行
            nearest = min(range(len(lines)), key=lambda i: abs(lines[i][0].cy - b.cy))
            if abs(lines[nearest][0].cy - b.cy) < lines[nearest][0].h:
                lines[nearest][1].append(b)
            else:
                lines.append((b, [b]))
        else:
            lines.append((b, [b]))
    lines.sort(key=lambda item: item[0].y0)
    return [members for _, members in lines]


def _merge_line(members: list[Box]) -> tuple[list[Box], float]:
    heights = sorted(b.h for b in members)
    line_h = float(np.percentile(heights, 90)) if heights else 1.0

    # 1. 合併上下疊的部件（二、三、i、j、？）
    boxes = sorted(members, key=lambda b: b.x0)
    merged: list[Box] = []
    for b in boxes:
        for i, m in enumerate(merged):
            if _horizontal_overlap(m, b) >= 0.4:
                merged[i] = m.union(b)
                break
        else:
            merged.append(b)
    merged.sort(key=lambda b: b.x0)

    # 2. 左右相鄰且合併後仍像一個方塊字的部件（川、好、明）
    changed = True
    while changed:
        changed = False
        for i in range(len(merged) - 1):
            a, b = merged[i], merged[i + 1]
            gap = b.x0 - a.x1
            u = a.union(b)
            narrow = a.w < 0.7 * line_h and b.w < 0.7 * line_h
            if narrow and gap < 0.35 * line_h and u.w <= 1.1 * line_h and u.h >= 0.6 * line_h:
                merged[i : i + 2] = [u]
                changed = True
                break
    return merged, line_h


def _score(box: Box, ink: np.ndarray, line_h: float) -> float:
    """方正、夠大、筆畫多的字最適合當風格參考。"""
    aspect = box.w / max(1, box.h)
    square = max(0.0, 1 - abs(np.log(max(aspect, 1e-3))) / np.log(2))
    size = min(1.0, box.h / max(1.0, line_h))
    # 筆畫複雜度：墨跡邊界長度相對字高
    edges = cv2.Canny(ink.astype(np.uint8) * 255, 50, 150)
    complexity = min(1.0, edges.sum() / 255 / max(1, box.h) / 12)
    return round(float(0.4 * square + 0.3 * size + 0.3 * complexity), 3)


def segment(image: Image.Image) -> Segmentation:
    gray = load_gray(image)
    ink, inverted = binarize(gray)
    result = Segmentation(ink=ink, inverted=inverted)
    boxes = _components(ink)
    if not boxes:
        return result
    for line_no, members in enumerate(_group_lines(boxes)):
        merged, line_h = _merge_line(members)
        ids = []
        for box in merged:
            crop = ink[box.y0 : box.y1, box.x0 : box.x1]
            if crop.sum() == 0:
                continue
            g = Glyph(index=len(result.glyphs), line=line_no, box=box, line_height=line_h, ink=crop)
            g.score = _score(box, crop, line_h)
            ids.append(g.index)
            result.glyphs.append(g)
        result.lines.append(ids)
    return result


def style_image(glyph: Glyph, size: int = STYLE_SIZE) -> Image.Image:
    """把單字做成 FontDiffuser 的風格參考圖：白底黑字、置中、正方形。

    以行高當作字身大小，讓英文字母保留和中文字之間的相對大小。
    """
    h, w = glyph.ink.shape
    em = max(glyph.line_height, w, h) / STYLE_FILL
    side = int(np.ceil(max(em, w + 2, h + 2)))
    canvas = np.full((side, side), 255, np.uint8)
    x, y = (side - w) // 2, (side - h) // 2
    canvas[y : y + h, x : x + w][glyph.ink] = 0
    return Image.fromarray(canvas).resize((size, size), Image.LANCZOS).convert("RGB")


def preview_image(glyph: Glyph, size: int = 72) -> Image.Image:
    """給網頁顯示用的單字縮圖。"""
    return style_image(glyph, size)
