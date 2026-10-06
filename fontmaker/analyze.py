"""分析圖中文字的視覺特色：粗細、對比、傾斜、字寬、圓潤度、工整度，並推測字體類型與字重。

這些數值不參與 AI 生成（FontDiffuser 直接從參考圖學風格），用途是：
1. 讓使用者了解圖中字型的特色；
2. 寫入字型檔的字重（usWeightClass）、斜體角度等中繼資料。
"""

import cv2
import numpy as np

from .segment import FILL_RATIO, Segmentation

WEIGHT_STOPS = ([0.02, 0.035, 0.05, 0.065, 0.08, 0.10, 0.12, 0.145, 0.17], [100, 200, 300, 400, 500, 600, 700, 800, 900])
WEIGHT_NAMES = {100: "Thin", 200: "ExtraLight", 300: "Light", 400: "Regular", 500: "Medium",
                600: "SemiBold", 700: "Bold", 800: "ExtraBold", 900: "Black"}


def _stroke_width(ink: np.ndarray) -> float:
    """平均筆畫寬度 ≈ 2 × 面積 ÷ 周長（對細長筆畫成立）。"""
    contours, _ = cv2.findContours(ink.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    return 2 * float(ink.sum()) / max(perimeter, 1.0)


def _contrast(ink: np.ndarray) -> float | None:
    """筆畫粗細對比：沿筆畫中線取寬度分佈的 85 / 15 百分位比。"""
    padded = np.pad(ink.astype(np.uint8), 2)
    dt = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
    ridge = (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8))) & (dt > 1.0)
    vals = dt[ridge]
    if len(vals) < 20:
        return None
    lo, hi = np.percentile(vals, [15, 85])
    return float(hi / max(lo, 0.5))


def _roundness(ink: np.ndarray, stroke: float) -> float | None:
    """邊角圓潤度（0–1）。

    沿輪廓找轉角，比較「小範圍內的轉向」與「整個轉角的轉向」：
    方角在一兩個像素內就轉完（比值接近 1），圓角是慢慢轉過去（比值較小）。
    筆畫太細時鋸齒會干擾判斷，回傳 None。
    """
    if stroke < 4:
        return None
    s = max(2, int(round(stroke * 0.15)))
    w = max(s * 2, int(round(stroke * 0.7)))
    contours, _ = cv2.findContours(ink.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    ratios = []
    for c in contours:
        p = c.reshape(-1, 2).astype(float)
        n = len(p)
        if n < 4 * w:
            continue
        idx = np.arange(n)
        d = p[(idx + s) % n] - p[(idx - s) % n]
        phi = np.arctan2(d[:, 1], d[:, 0])

        def turn(k: int) -> np.ndarray:
            return np.abs((phi[(idx + k) % n] - phi[(idx - k) % n] + np.pi) % (2 * np.pi) - np.pi)

        big, small = turn(w), turn(s)
        peaks = (big > np.radians(60)) & (big >= np.roll(big, 1)) & (big >= np.roll(big, -1))
        for i in np.where(peaks)[0]:
            window = (i + np.arange(-w, w + 1)) % n
            ratios.append(small[window].max() / big[i])
    if len(ratios) < 3:
        return None
    return float(np.clip((0.85 - np.median(ratios)) / 0.25, 0, 1))


def _slant(ink: np.ndarray) -> float:
    """找出讓直筆最「直」的水平錯切角度（度，正值為向右傾）。"""
    h, w = ink.shape
    if h < 8 or w < 8:
        return 0.0
    src = ink.astype(np.float32)
    best_angle, best_score = 0.0, -1.0
    pad = int(h * np.tan(np.radians(25))) + 1
    for angle in np.arange(-25, 25.5, 1.0):
        shear = np.tan(np.radians(angle))
        m = np.float32([[1, shear, pad - shear * h], [0, 1, 0]])
        warped = cv2.warpAffine(src, m, (w + 2 * pad, h))
        score = float((warped.sum(axis=0) ** 2).sum())
        if score > best_score:
            best_angle, best_score = float(angle), score
    return best_angle


def _describe(value: float, stops: list[tuple[float, str]]) -> str:
    for limit, label in stops:
        if value < limit:
            return label
    return stops[-1][1]


def analyze(seg: Segmentation) -> dict:
    glyphs = seg.glyphs
    if not glyphs:
        return {"ok": False, "message": "圖片中找不到文字，請換一張對比較清楚的圖片。"}

    good = [g for g in glyphs if g.score >= 0.5] or glyphs
    strokes, contrasts, rounds, ratios = [], [], [], []
    for g in good:
        sw = _stroke_width(g.ink)
        strokes.append(sw / (g.line_height / FILL_RATIO))
        ratios.append(g.box.w / max(1, g.box.h))
        c = _contrast(g.ink)
        if c is not None:
            contrasts.append(c)
        r = _roundness(g.ink, sw)
        if r is not None:
            rounds.append(r)

    stroke_em = float(np.median(strokes))
    stroke_cv = float(np.std(strokes) / max(np.mean(strokes), 1e-6))
    contrast = float(np.median(contrasts)) if contrasts else 1.0
    roundness = float(np.median(rounds)) if rounds else None
    aspect = float(np.median(ratios))

    # 傾斜：以每一行整體估計，依行長加權
    slants, weights = [], []
    for ids in seg.lines:
        if not ids:
            continue
        box = glyphs[ids[0]].box
        for i in ids[1:]:
            box = box.union(glyphs[i].box)
        slants.append(_slant(seg.ink[box.y0 : box.y1, box.x0 : box.x1]))
        weights.append(len(ids))
    slant = float(np.average(slants, weights=weights)) if slants else 0.0

    # 工整度：方塊字底線的起伏 + 筆畫粗細的變異
    jitter = []
    for ids in seg.lines:
        bottoms = [glyphs[i].box.y1 / glyphs[i].line_height for i in ids if glyphs[i].score >= 0.5]
        if len(bottoms) >= 3:
            jitter.append(float(np.std(bottoms)))
    baseline_jitter = float(np.median(jitter)) if jitter else 0.0
    irregularity = float(np.clip(stroke_cv * 1.5 + baseline_jitter * 4, 0, 1))

    weight = int(round(np.interp(stroke_em, *WEIGHT_STOPS) / 100) * 100)

    if irregularity > 0.45 or abs(slant) > 8:
        style = "手寫／書法風格"
        style_note = "字形不規則或明顯傾斜，較接近手寫或楷書。"
    elif contrast > 2.2:
        style = "明體／宋體風格"
        style_note = "橫細直粗、粗細對比明顯。"
    elif roundness is not None and roundness > 0.6:
        style = "圓體風格"
        style_note = "筆畫粗細均勻、轉角與筆端圓潤。"
    else:
        style = "黑體風格"
        style_note = "筆畫粗細均勻、轉角方正。"

    features = [
        {"key": "stroke", "label": "筆畫粗細", "value": round(stroke_em * 100, 1), "unit": "% 字身",
         "text": _describe(stroke_em, [(0.035, "極細"), (0.055, "細"), (0.085, "中等"), (0.12, "粗"), (9, "極粗")])},
        {"key": "contrast", "label": "粗細對比", "value": round(contrast, 2), "unit": "倍",
         "text": _describe(contrast, [(1.5, "均勻"), (2.2, "略有對比"), (9e9, "對比強烈")])},
        {"key": "slant", "label": "傾斜角度", "value": round(slant, 1), "unit": "°",
         "text": "正體" if abs(slant) < 3 else ("向右傾" if slant > 0 else "向左傾")},
        {"key": "aspect", "label": "字寬比（寬÷高）", "value": round(aspect, 2), "unit": "",
         "text": _describe(aspect, [(0.85, "長體"), (1.1, "方正"), (9e9, "扁平")])},
        {"key": "roundness", "label": "邊角圓潤度", "value": None if roundness is None else round(roundness * 100),
         "unit": "/ 100", "text": "字太小，無法判斷" if roundness is None else
         _describe(roundness, [(0.3, "方正銳利"), (0.6, "略圓"), (9, "圓潤")])},
        {"key": "irregularity", "label": "手寫感", "value": round(irregularity * 100), "unit": "/ 100",
         "text": _describe(irregularity, [(0.2, "工整"), (0.45, "略有變化"), (9, "明顯手寫感")])},
    ]
    return {
        "ok": True,
        "glyph_count": len(glyphs),
        "line_count": len(seg.lines),
        "inverted": seg.inverted,
        "style": style,
        "style_note": style_note,
        "weight": weight,
        "weight_name": WEIGHT_NAMES[weight],
        "slant": slant,
        "features": features,
    }
