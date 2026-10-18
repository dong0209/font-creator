"""字集定義。

中文字集取自 Big5 編碼的「常用字」與「次常用字」區段，這兩區即依教育部常用／次常用國字表編排，
由 Python 內建的 big5 codec 直接解碼產生，不需要另外下載字表。
"""

from functools import lru_cache

# Big5 第一字面（常用字）與第二字面（次常用字）的碼位範圍
BIG5_LEVEL1 = (0xA440, 0xC67E)
BIG5_LEVEL2 = (0xC940, 0xF9D5)

ASCII_PRINTABLE = "".join(chr(c) for c in range(0x21, 0x7F))

CJK_PUNCTUATION = (
    "，。、；：？！…—～·‧"
    "「」『』（）《》〈〉【】〔〕"
    "﹁﹂﹃﹄＂＇"
    "　"
)


def _big5_range(start: int, end: int) -> str:
    chars = []
    for hi in range(start >> 8, (end >> 8) + 1):
        for lo in list(range(0x40, 0x7F)) + list(range(0xA1, 0xFF)):
            code = (hi << 8) | lo
            if start <= code <= end:
                try:
                    chars.append(bytes([hi, lo]).decode("big5"))
                except UnicodeDecodeError:
                    pass
    return "".join(chars)


@lru_cache(maxsize=None)
def common() -> str:
    """常用字 5,401 字。"""
    return _big5_range(*BIG5_LEVEL1)


@lru_cache(maxsize=None)
def less_common() -> str:
    """次常用字 7,652 字。"""
    return _big5_range(*BIG5_LEVEL2)


PRESETS = {
    "common": "常用字（5,401 字，建議）",
    "common+less": "常用＋次常用字（13,053 字）",
    "custom": "自訂字表",
}


def build_charset(preset: str, custom: str = "", include_latin: bool = True) -> list[str]:
    """依選項組出要生成的字元清單（去重、保留順序；空白字元另外處理不在此列）。"""
    if preset == "common":
        base = common()
    elif preset == "common+less":
        base = common() + less_common()
    elif preset == "custom":
        base = ""
    else:
        raise ValueError(f"未知的字集：{preset}")

    parts = []
    if include_latin:
        parts.append(ASCII_PRINTABLE)
    parts.append(CJK_PUNCTUATION.replace("　", ""))
    parts.append(base)
    parts.append(custom)

    seen = set()
    out = []
    for ch in "".join(parts):
        if ch.isspace() or ch in seen:
            continue
        seen.add(ch)
        out.append(ch)
    return out
