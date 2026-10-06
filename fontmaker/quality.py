"""生成結果的品質評估：多參考字時挑出最好的候選，並計算每個字的筆畫粗細修正。"""

import cv2
import numpy as np

from . import vectorize

GRID = 32


def binarize(image) -> np.ndarray:
    gray = np.array(image.convert("L")) if hasattr(image, "convert") else image
    return gray < 128


def _shape_vector(binary: np.ndarray) -> np.ndarray | None:
    """把墨跡裁到外框、縮成 32×32 再模糊，用來比較結構（與位置、大小無關）。"""
    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        return None
    crop = binary[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].astype(np.float32)
    h, w = crop.shape
    side = max(h, w)
    square = np.zeros((side, side), np.float32)
    square[(side - h) // 2 : (side - h) // 2 + h, (side - w) // 2 : (side - w) // 2 + w] = crop
    small = cv2.resize(square, (GRID, GRID), interpolation=cv2.INTER_AREA)
    v = cv2.GaussianBlur(small, (0, 0), 1.2).ravel()
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def structure_similarity(gen: np.ndarray, content: np.ndarray) -> float:
    """生成字與內容字的結構相似度（-1–1）：用來抓缺筆、多筆、糊成一團的失敗結果。"""
    a, b = _shape_vector(gen), _shape_vector(content)
    if a is None or b is None:
        return -1.0
    return float(np.dot(a, b))


def noise_fraction(binary: np.ndarray, min_area: int = 9) -> float:
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    total = binary.sum()
    if total == 0:
        return 1.0
    small = sum(stats[i, cv2.CC_STAT_AREA] for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] < min_area)
    return float(small / total)


def content_binary(content_image, size: int = 96) -> np.ndarray:
    gray = np.array(content_image.convert("L").resize((size, size)))
    return gray < 128


def pick_best(candidates: list, content_image) -> tuple[int, list[float]]:
    """從同一個字的多個候選中挑最好的。回傳 (索引, 各候選分數)。

    分數 = 結構相似度 − 粗細偏離（相對於其他候選的中位數）− 雜點比例。
    """
    if len(candidates) == 1:
        return 0, [0.0]
    content = content_binary(content_image)
    bins = [binarize(c) for c in candidates]
    widths = [vectorize.stroke_width(b) if b.any() else 0.0 for b in bins]
    valid = [w for w in widths if w > 0]
    median_w = float(np.median(valid)) if valid else 1.0
    scores = []
    for b, w in zip(bins, widths):
        if w <= 0:
            scores.append(-9.0)
            continue
        sim = structure_similarity(b, content)
        weight_dev = abs(np.log(w / median_w))
        scores.append(sim - 0.5 * weight_dev - 2.0 * noise_fraction(b))
    return int(np.argmax(scores)), [round(s, 4) for s in scores]


def weight_corrections(gen_widths: dict[str, float], content_widths: dict[str, float],
                       tolerance: float = 0.08, max_change: float = 0.3) -> dict[str, float]:
    """計算每個字要外擴／內縮多少像素，讓整套字粗細一致。

    不是把所有字拉到同一粗細（筆畫多的字本來就該細一點），
    而是讓「生成粗細 ÷ 內容字粗細」的比例一致：以全體中位數為目標。
    """
    ratios = [gen_widths[c] / content_widths[c] for c in gen_widths
              if gen_widths[c] > 0 and content_widths.get(c, 0) > 0]
    if len(ratios) < 5:
        return {}
    target_ratio = float(np.median(ratios))
    out = {}
    for c, w in gen_widths.items():
        cw = content_widths.get(c, 0)
        if w <= 0 or cw <= 0:
            continue
        expected = cw * target_ratio
        if abs(w - expected) / expected <= tolerance:
            continue
        change = float(np.clip(expected - w, -max_change * w, max_change * w))
        out[c] = change / 2  # 兩側各外擴／內縮一半
    return out
