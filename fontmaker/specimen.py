"""字型樣張：展示圖（PNG）與全字表（PDF）。"""

from pathlib import Path

from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

SAMPLE_ZH = "永和九年，歲在癸丑，暮春之初，會於會稽山陰之蘭亭。"
SAMPLE_EN = "The quick brown fox jumps over the lazy dog 0123456789"
BIG = "永國鬱靈"
INK, MUTED, LINE, BG = (24, 24, 27), (110, 110, 118), (224, 222, 217), (255, 255, 255)


def _cmap(font_path: str) -> set[int]:
    return set((TTFont(font_path, lazy=True).getBestCmap() or {}).keys())


def _has(font_path: str, text: str) -> str:
    """只留字型裡有的字，避免出現方框。"""
    cmap = _cmap(font_path)
    return "".join(c for c in text if c.isspace() or ord(c) in cmap)


def _sample(font_path: str, preferred: str, length: int) -> str:
    """優先用範例句；字型裡缺太多字時（例如只生成了自訂的少數字），改用字型裡實際有的中文字。"""
    cmap = _cmap(font_path)
    kept = "".join(c for c in preferred if ord(c) in cmap)
    if len(kept) >= 0.6 * len(preferred):
        return kept
    cjk = [chr(c) for c in sorted(cmap) if 0x4E00 <= c <= 0x9FFF]
    return "".join(cjk[:length]) or kept


def render_specimen(family: str, weights: list[dict], out_path: Path, ui_font: str, width: int = 1600) -> Path:
    """weights：[{"style": "Regular", "weight": 400, "ttf": 路徑}, ...]，依字重排序。

    ui_font：標題與說明文字用的字型（一定要有中文，例如內容字型）。
    """
    regular = min(weights, key=lambda w: abs(w["weight"] - 400))["ttf"]
    pad = 64
    rows = []  # (高度, 繪製函式)

    def text_row(font_path, size, text, color=INK, gap=0.45):
        f = ImageFont.truetype(font_path, size)
        h = int(size * (1 + gap))
        return h, lambda d, y: d.text((pad, y), _has(font_path, text), font=f, fill=color)

    title = ImageFont.truetype(ui_font, 64)
    note = ImageFont.truetype(ui_font, 26)
    rows.append((100, lambda d, y: d.text((pad, y), family, font=title, fill=INK)))
    rows.append((70, lambda d, y: d.text((pad, y), f"{len(weights)} 種字重・此字型禁止用於商業用途（Non-Commercial）",
                                         font=note, fill=MUTED)))
    big = ImageFont.truetype(regular, 220)
    big_text = _sample(regular, BIG, 4)
    rows.append((300, lambda d, y: d.text((pad, y), big_text, font=big, fill=INK)))
    zh = _sample(regular, SAMPLE_ZH, 24)
    for w in weights:
        label = ImageFont.truetype(ui_font, 22)
        sample = ImageFont.truetype(w["ttf"], 44)

        def draw_weight(d, y, w=w, label=label, sample=sample):
            d.line((pad, y, width - pad, y), fill=LINE, width=2)
            d.text((pad, y + 14), f"{w['style']}  {w['weight']}", font=label, fill=MUTED)
            d.text((pad + 220, y + 6), _has(w["ttf"], zh), font=sample, fill=INK)
            d.text((pad + 220, y + 66), _has(w["ttf"], SAMPLE_EN), font=sample, fill=INK)

        rows.append((136, draw_weight))
    height = pad * 2 + sum(h for h, _ in rows)
    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)
    y = pad
    for h, draw in rows:
        draw(d, y)
        y += h
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def render_glyph_chart(family: str, font_path: str, out_path: Path, ui_font: str,
                       cols: int = 18, rows: int = 24) -> Path:
    """全字表（A4 直式、150dpi），每格一個字，依字碼排序。"""
    font = TTFont(font_path, lazy=True)
    chars = sorted(chr(c) for c in (font.getBestCmap() or {}) if c > 0x20 and c != 0x3000)
    page_w, page_h, margin, header = 1240, 1754, 70, 90
    cell = min((page_w - 2 * margin) // cols, (page_h - 2 * margin - header) // rows)
    glyph = ImageFont.truetype(font_path, int(cell * 0.7))
    title = ImageFont.truetype(ui_font, 34)
    small = ImageFont.truetype(ui_font, 18)
    per_page = cols * rows
    pages = []
    total = max(1, -(-len(chars) // per_page))
    for p in range(total):
        img = Image.new("RGB", (page_w, page_h), BG)
        d = ImageDraw.Draw(img)
        d.text((margin, margin), family, font=title, fill=INK)
        d.text((page_w - margin, margin + 12), f"{len(chars):,} 字・{p + 1} / {total}",
               font=small, fill=MUTED, anchor="ra")
        top = margin + header
        for k, ch in enumerate(chars[p * per_page : (p + 1) * per_page]):
            r, c = divmod(k, cols)
            x, y = margin + c * cell, top + r * cell
            d.rectangle((x, y, x + cell, y + cell), outline=LINE)
            d.text((x + cell / 2, y + cell / 2), ch, font=glyph, fill=INK, anchor="mm")
        pages.append(img)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(out_path, "PDF", resolution=150, save_all=True, append_images=pages[1:])
    return out_path
