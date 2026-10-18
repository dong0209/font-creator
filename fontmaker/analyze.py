"""分析圖中文字的視覺特色：粗細、對比、傾斜、字寬、圓潤度、工整度，並推測字體類型與字重。

這些數值不參與 AI 生成（FontDiffuser 直接從參考圖學風格），用途是：
1. 讓使用者了解圖中字型的特色；
2. 寫入字型檔的字重（usWeightClass）、斜體角度等中繼資料。
"""

import cv2
import numpy as np

from .segment import FILL_RATIO, Segmentation

WEIGHT_STOPS = ([0.02, 0.035, 0.05, 0.065, 0.08, 0.10, 0.12, 0.145, 0.17], [100, 200, 300, 400, 500, 600, 700, 800, 900])
# 清晰的圖（含反鋸齒與二值化前的輕微平滑）模糊估計約 1.0–1.1
BLURRY = 1.2
WEIGHT_NAMES = {100: "Thin", 200: "ExtraLight", 300: "Light", 400: "Regular", 500: "Medium",
                600: "SemiBold", 700: "Bold", 800: "ExtraBold", 900: "Black"}


def _stroke_width(ink: np.ndarray) -> float:
    """平均筆畫寬度 ≈ 2 × 面積 ÷ 周長（對細長筆畫成立）。"""
    contours, _ = cv2.findContours(ink.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    return 2 * float(ink.sum()) / max(perimeter, 1.0)


def _runs(ink: np.ndarray, axis: int) -> np.ndarray:
    """沿某個方向的連續墨跡長度。axis=0：每一欄的垂直長度；axis=1：每一列的水平長度。"""
    a = ink if axis == 0 else ink.T
    d = np.diff(np.pad(a.astype(np.int8), ((1, 1), (0, 0))), axis=0)
    starts, ends = np.nonzero(d.T == 1), np.nonzero(d.T == -1)
    return (ends[1] - starts[1]).astype(np.float64)


def _contrast(ink: np.ndarray) -> float | None:
    """粗細對比：直畫粗細 ÷ 橫畫粗細。

    橫畫的粗細 ≈ 垂直方向的短墨跡段、直畫的粗細 ≈ 水平方向的短墨跡段，各取第 25 百分位。
    黑體／圓體約 1.1–1.3，明體（橫細直粗）約 2 以上。
    """
    v, h = _runs(ink, 0), _runs(ink, 1)
    v, h = v[v >= 2], h[h >= 2]
    if len(v) < 10 or len(h) < 10:
        return None
    return float(np.percentile(h, 25) / max(np.percentile(v, 25), 1.0))


def _edge_angles(ink: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """筆畫邊緣的角度：回傳 (直畫邊緣偏角, 權重, 橫畫邊緣偏角, 權重)，單位為度。"""
    f = cv2.GaussianBlur(ink.astype(np.float32), (0, 0), 1.0)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    strong = mag > mag.max() * 0.2 if mag.max() > 0 else mag > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        dv = np.degrees(np.arctan(gy / gx))  # 直畫邊緣（梯度近水平）
        dh = np.degrees(np.arctan(gx / gy))  # 橫畫邊緣（梯度近垂直）
    sv = strong & (np.abs(dv) < 25)
    sh = strong & (np.abs(dh) < 25)
    return dv[sv], mag[sv], dh[sh], mag[sh]


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cum, cum[-1] / 2)])


def _roundness(ink: np.ndarray, stroke: float, blur: float = 0.0) -> float | None:
    """邊角圓潤度（0–1）。

    沿輪廓找轉角，比較「小範圍內的轉向」與「整個轉角的轉向」：
    方角在一兩個像素內就轉完（比值接近 1），圓角是慢慢轉過去（比值較小）。
    筆畫太細時鋸齒會干擾判斷，回傳 None。
    """
    if stroke < 4:
        return None
    s = max(2, int(round(stroke * 0.15)))
    # 照片模糊會把方角磨圓（半徑約為模糊 σ）。低解析度時這跟真正的圓角分不出來，直接不判斷；
    # 解析度夠時，把量測窗口放大到超過模糊造成的圓角
    if blur > BLURRY:
        if stroke < 12:
            return None
        s = max(s, int(round(stroke * 0.2)), int(round(blur * 2.0)))
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
    """傾斜角度（度，正值為向右傾）：直畫邊緣角度的加權中位數。"""
    dv, wv, _, _ = _edge_angles(ink)
    return _weighted_median(dv, wv)


def _rise(ink: np.ndarray) -> float:
    """橫畫上揚程度：橫畫邊緣中「右上揚」比「右下垂」多出的比例。楷書約 0.05 以上，印刷黑體／明體接近 0。"""
    _, _, dh, wh = _edge_angles(ink)
    if wh.sum() == 0:
        return 0.0
    up = wh[(dh > 2) & (dh < 15)].sum()
    down = wh[(dh < -2) & (dh > -15)].sum()
    return float((up - down) / wh.sum())


def _describe(value: float, stops: list[tuple[float, str]]) -> str:
    for limit, label in stops:
        if value < limit:
            return label
    return stops[-1][1]


def _subset_ink(seg: Segmentation, ids: list[int]) -> np.ndarray:
    """只含指定字的墨跡（其他字留白），用來分別估計各組的傾斜與上揚。"""
    mask = np.zeros_like(seg.ink)
    for i in ids:
        g = seg.glyphs[i]
        mask[g.box.y0 : g.box.y1, g.box.x0 : g.box.x1] |= g.ink
    return mask


def analyze(seg: Segmentation, glyph_ids: list[int] | None = None) -> dict:
    """分析整張圖，或只分析 glyph_ids 指定的字（一個字型分組）。"""
    if glyph_ids is None:
        glyph_ids = list(range(len(seg.glyphs)))
    id_set = set(glyph_ids)
    glyphs = [seg.glyphs[i] for i in glyph_ids]
    lines = [[i for i in ids if i in id_set] for ids in seg.lines]
    lines = [ids for ids in lines if ids]
    if not glyphs:
        return {"ok": False, "message": "圖片中找不到文字，請換一張對比較清楚的圖片。"}

    # 只用接近完整字大小的字（排除標點等小符號）
    full = [g for g in glyphs if max(g.box.w, g.box.h) >= 0.7 * g.line_height]
    good = [g for g in full if g.score >= 0.5] or full or glyphs
    strokes, contrasts, rounds, ratios, horizontals = [], [], [], [], []
    for g in good:
        sw = _stroke_width(g.ink)
        strokes.append(sw / (g.line_height / FILL_RATIO))
        ratios.append(g.box.w / max(1, g.box.h))
        c = _contrast(g.ink)
        if c is not None:
            contrasts.append(c)
        v = _runs(g.ink, 0)
        v = v[v >= 2]
        if len(v) >= 10:  # 橫畫粗細（佔字身比例）
            horizontals.append(float(np.percentile(v, 25)) / (g.line_height / FILL_RATIO))
        r = _roundness(g.ink, sw, seg.blur)
        if r is not None:
            rounds.append(r)

    stroke_em = float(np.median(strokes))
    stroke_cv = float(np.std(strokes) / max(np.mean(strokes), 1e-6))
    contrast = float(np.median(contrasts)) if contrasts else 1.0
    horizontal = float(np.median(horizontals)) if horizontals else 1.0
    roundness = float(np.median(rounds)) if rounds else None
    aspect = float(np.median(ratios))

    ink = _subset_ink(seg, glyph_ids)
    slant = _slant(ink)
    rise = _rise(ink)
    glyph_px = float(np.median([g.box.h for g in good]))

    # 工整度：方塊字底線的起伏 + 筆畫粗細的變異
    jitter = []
    for ids in lines:
        bottoms = [seg.glyphs[i].box.y1 / seg.glyphs[i].line_height for i in ids if seg.glyphs[i] in good]
        if len(bottoms) >= 3:
            jitter.append(float(np.std(bottoms)))
    baseline_jitter = float(np.median(jitter)) if jitter else 0.0
    irregularity = float(np.clip(stroke_cv * 1.5 + baseline_jitter * 4, 0, 1))

    weight = int(round(np.interp(stroke_em, *WEIGHT_STOPS) / 100) * 100)

    # 英文字母比方塊字窄得多；大多數字都很窄時，視為以英文為主
    narrow = sum(1 for g in glyphs if g.box.w < 0.75 * g.line_height)
    latin_heavy = narrow / len(glyphs) > 0.5

    if latin_heavy:
        if irregularity > 0.45 or abs(slant) > 8:
            style, style_note = "手寫／草寫風格（英文）", "以英文字母判斷：字形不規則或明顯傾斜。"
        elif contrast > 1.4 and horizontal < 0.048:  # 明體：橫畫明顯比直畫細，而且橫畫本身就很細
            style, style_note = "襯線風格（英文）", "以英文字母判斷：直粗橫細，接近襯線（Serif）字型。"
        else:
            style, style_note = "無襯線風格（英文）", "以英文字母判斷：筆畫粗細均勻，接近無襯線（Sans）字型。"
    elif irregularity > 0.45 or abs(slant) > 8:
        style = "手寫／書法風格"
        style_note = "字形不規則或明顯傾斜，較接近手寫。"
    elif rise > 0.035:
        style = "楷書風格"
        style_note = "橫畫向右上揚、筆畫有提按變化，接近楷書或手寫。"
    elif contrast > 1.4 and horizontal < 0.048:  # 明體：橫畫明顯比直畫細，而且橫畫本身就很細
        style = "明體／宋體風格"
        style_note = "橫細直粗、粗細對比明顯。"
    elif roundness is not None and roundness > (0.65 if seg.blur > BLURRY else 0.4):
        style = "圓體風格"
        style_note = "筆畫粗細均勻、轉角與筆端圓潤。"
    else:
        style = "黑體風格"
        style_note = "筆畫粗細均勻、轉角方正。"

    features = [
        {"key": "stroke", "label": "筆畫粗細", "value": round(stroke_em * 100, 1), "unit": "% 字身",
         "text": _describe(stroke_em, [(0.035, "極細"), (0.055, "細"), (0.085, "中等"), (0.12, "粗"), (9, "極粗")])},
        {"key": "contrast", "label": "粗細對比（直÷橫）", "value": round(contrast, 2), "unit": "倍",
         "text": _describe(contrast, [(1.3, "均勻"), (1.7, "略有對比"), (9e9, "橫細直粗")])},
        {"key": "rise", "label": "橫畫上揚", "value": round(rise * 100), "unit": "/ 100",
         "text": _describe(rise, [(0.02, "水平"), (0.035, "略上揚"), (9, "明顯上揚")])},
        {"key": "slant", "label": "傾斜角度", "value": round(slant, 1), "unit": "°",
         "text": "正體" if abs(slant) < 3 else ("向右傾" if slant > 0 else "向左傾")},
        {"key": "aspect", "label": "字寬比（寬÷高）", "value": round(aspect, 2), "unit": "",
         "text": _describe(aspect, [(0.85, "長體"), (1.1, "方正"), (9e9, "扁平")])},
        {"key": "roundness", "label": "邊角圓潤度", "value": None if roundness is None else round(roundness * 100),
         "unit": "/ 100", "text": "字太小或太模糊，無法判斷" if roundness is None else
         _describe(roundness, [(0.25, "方正銳利"), (0.4, "略圓"), (9, "圓潤")])},
        {"key": "irregularity", "label": "手寫感", "value": round(irregularity * 100), "unit": "/ 100",
         "text": _describe(irregularity, [(0.2, "工整"), (0.45, "略有變化"), (9, "明顯手寫感")])},
    ]
    warnings = []
    if latin_heavy:
        warnings.append("圖中主要是英文字母。AI 模型以中文字訓練，用英文當風格參考效果較差，建議上傳含中文字的圖片。")
    if glyph_px < 40:
        warnings.append(f"字太小（每字約 {glyph_px:.0f}px），分析數值與 AI 生成效果都會變差，建議改用每字 60px 以上的圖片。")
    elif glyph_px < 60:
        warnings.append(f"字偏小（每字約 {glyph_px:.0f}px），若有更高解析度的圖片效果會更好。")
    return {
        "ok": True,
        "warnings": warnings,
        "glyph_px": round(glyph_px),
        "glyph_count": len(glyphs),
        "line_count": len(lines),
        "inverted": seg.inverted,
        "style": style,
        "style_note": style_note,
        "weight": weight,
        "weight_name": WEIGHT_NAMES[weight],
        "slant": slant,
        "features": features,
    }


def _line_features(seg: Segmentation, ids: list[int]) -> dict | None:
    full = [seg.glyphs[i] for i in ids if max(seg.glyphs[i].box.w, seg.glyphs[i].box.h) >= 0.7 * seg.glyphs[i].line_height]
    if not full:
        return None
    height = float(np.median([g.box.h for g in full]))
    strokes = [_stroke_width(g.ink) / max(g.box.h, 1) for g in full]
    contrasts = [c for c in (_contrast(g.ink) for g in full) if c is not None]
    return {
        "height": height,
        "stroke": float(np.median(strokes)),
        "contrast": float(np.median(contrasts)) if contrasts else 1.2,
        "rise": _rise(_subset_ink(seg, [g.index for g in full])),
        "y": float(np.mean([g.box.cy for g in full])),
    }


def _same_font(a: dict, b: dict) -> bool:
    ratio = lambda x, y: max(x, y) / max(min(x, y), 1e-6)  # noqa: E731
    return (ratio(a["height"], b["height"]) <= 1.3
            and ratio(a["stroke"], b["stroke"]) <= 1.3
            and abs(a["contrast"] - b["contrast"]) <= 0.35
            and abs(a["rise"] - b["rise"]) <= 0.03)


def group_glyphs(seg: Segmentation) -> list[list[int]]:
    """把圖中的字依字型分組：同一行視為同一字型，字高、筆畫粗細比例、粗細對比、橫畫上揚都相近的行歸成一組。

    只有標點等小符號的行，併到上下最近的一組。回傳各組的字索引，依在圖中由上到下排序。
    """
    feats = [(ids, _line_features(seg, ids)) for ids in seg.lines if ids]
    real = [(ids, f) for ids, f in feats if f is not None]
    if not real:
        return [list(range(len(seg.glyphs)))] if seg.glyphs else []

    parent = list(range(len(real)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(real)):
        for j in range(i + 1, len(real)):
            if _same_font(real[i][1], real[j][1]):
                parent[find(j)] = find(i)

    groups: dict[int, list[int]] = {}
    centers: dict[int, list[float]] = {}
    for k, (ids, f) in enumerate(real):
        groups.setdefault(find(k), []).extend(ids)
        centers.setdefault(find(k), []).append(f["y"])
    for ids, f in feats:
        if f is None:  # 只有小符號的行：併到最近的一組
            y = float(np.mean([seg.glyphs[i].box.cy for i in ids]))
            nearest = min(centers, key=lambda r: min(abs(c - y) for c in centers[r]))
            groups[nearest].extend(ids)
    ordered = sorted(groups.values(), key=lambda ids: min(seg.glyphs[i].box.y0 for i in ids))
    return [sorted(ids) for ids in ordered]


def stroke_for_weight(weight: float) -> float:
    """字重 → 典型筆畫粗細（字身比例），WEIGHT_STOPS 的反查。"""
    strokes, weights = WEIGHT_STOPS
    return float(np.interp(weight, weights, strokes))


def weight_variants(base: int) -> list[tuple[str, int]]:
    """由分析出的字重衍生一組字重（常見組合，例如 400 → Light 300／Regular 400／Bold 700）。

    筆畫加粗或變細太多形狀會崩，所以把粗細倍率限制在 0.55–1.9 倍之間。
    """
    base = int(np.clip(round(base / 100) * 100, 100, 900))
    lighter = base - (100 if base <= 400 else 200)
    heavier = base + (300 if base <= 400 else 200)
    ref = stroke_for_weight(base)
    while lighter < base and stroke_for_weight(lighter) / ref < 0.55:
        lighter += 100
    while heavier > base and stroke_for_weight(heavier) / ref > 1.9:
        heavier -= 100
    weights = sorted({int(np.clip(w, 100, 900)) for w in (lighter, base, heavier)})
    return [(WEIGHT_NAMES[w], w) for w in weights]
