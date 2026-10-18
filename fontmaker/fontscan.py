"""掃描本機可用的字型（fonts/ 資料夾＋作業系統字型），供內容字型與英數字比對使用。

結果依檔案修改時間快取在 work/fontscan.json，第二次啟動幾乎不花時間。
"""

import json
import os
import re
import sys
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

from fontTools.ttLib import TTCollection, TTFont

from . import charsets, config

CACHE_VERSION = 2
ASCII_REQUIRED = set(range(0x21, 0x7F))
# 宋體／明體類字型最接近 FontDiffuser 訓練時的內容字型
SONG_PATTERN = re.compile(r"song|sung|ming|宋|明|serif|kaixin", re.IGNORECASE)
PREFERRED_CONTENT = re.compile(r"kaixinsong|开心宋|開心宋", re.IGNORECASE)
# 點陣、示範用或符號字型不適合當內容字型
EXCLUDED_CONTENT = re.compile(r"unifont|sample|pixel|bitmap|emoji|symbol|dingbat|\bLast ?Resort", re.IGNORECASE)


@dataclass
class FaceInfo:
    id: str  # "<path>#<index>"
    path: str
    index: int
    family: str
    style: str
    cjk_coverage: float  # 對常用字的涵蓋率
    ascii: bool
    italic_angle: float
    user: bool  # 是否放在 fonts/ 資料夾（使用者自己放的優先）
    features: dict | None = field(default=None)  # 英數字特徵，需要時才計算

    @property
    def label(self) -> str:
        name = f"{self.family} {self.style}".strip() if self.style not in ("Regular", "") else self.family
        return name


def system_font_dirs() -> list[Path]:
    home = Path.home()
    if sys.platform == "darwin":
        dirs = [Path("/System/Library/Fonts"), Path("/Library/Fonts"), home / "Library/Fonts"]
        # macOS 可另外下載的字型（例如宋體-繁、蘭亭黑）放在 AssetsV2 底下
        dirs += sorted(Path("/System/Library/AssetsV2").glob("com_apple_MobileAsset_Font*"))
    elif os.name == "nt":
        windir = Path(os.environ.get("WINDIR", "C:/Windows"))
        dirs = [windir / "Fonts", Path(os.environ.get("LOCALAPPDATA", home)) / "Microsoft/Windows/Fonts"]
    else:
        dirs = [Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), home / ".local/share/fonts", home / ".fonts"]
    extra = os.environ.get("FONTMAKER_EXTRA_FONT_DIRS", "")
    dirs += [Path(p) for p in extra.split(os.pathsep) if p]
    return [d for d in dirs if d.is_dir()]


def _font_files(directory: Path) -> list[Path]:
    out = []
    for root, _, files in os.walk(directory):
        for name in files:
            if Path(name).suffix.lower() in config.FONT_EXTS:
                out.append(Path(root) / name)
    return out


def _names(font: TTFont) -> tuple[str, str]:
    name = font["name"]
    family = None
    for lang in (0x0404, 0x0C04, 0x0804, 0x0409):  # 繁中、港、簡中、英
        rec = name.getName(16, 3, 1, lang) or name.getName(1, 3, 1, lang)
        if rec:
            family = rec.toUnicode()
            break
    family = family or name.getDebugName(16) or name.getDebugName(1) or "?"
    style = name.getDebugName(17) or name.getDebugName(2) or ""
    return family.strip(), style.strip()


def _inspect(path: Path, user: bool) -> list[FaceInfo]:
    common = {ord(c) for c in charsets.common()}
    faces = []
    try:
        if path.suffix.lower() == ".ttc":
            fonts = TTCollection(str(path), lazy=True).fonts
        else:
            fonts = [TTFont(str(path), lazy=True)]
    except Exception:  # noqa: BLE001 - 壞掉或不支援的字型直接略過
        return []
    for index, font in enumerate(fonts):
        try:
            cmap = set((font.getBestCmap() or {}).keys())
            family, style = _names(font)
            angle = float(getattr(font.get("post"), "italicAngle", 0) or 0)
        except Exception:  # noqa: BLE001
            continue
        faces.append(FaceInfo(
            id=f"{path}#{index}", path=str(path), index=index, family=family, style=style,
            cjk_coverage=round(len(common & cmap) / len(common), 4),
            ascii=ASCII_REQUIRED <= cmap, italic_angle=angle, user=user,
        ))
    return faces


class FontScanner:
    def __init__(self, cache_path: Path | None = None):
        self.cache_path = cache_path or (config.WORK_DIR / "fontscan.json")
        self._faces: dict[str, FaceInfo] = {}
        self._lock = threading.Lock()
        self._mtimes: dict[str, float] = {}
        self.ready = threading.Event()  # 字型清單掃描完成
        self.latin_ready = threading.Event()  # 英數字特徵計算完成

    def start(self) -> None:
        def run():
            self.scan()
            self.compute_latin_features()

        threading.Thread(target=run, daemon=True, name="fontscan").start()

    def compute_latin_features(self) -> None:
        from .latin import font_features

        todo = [f for f in self.latin_fonts() if f.features is None]
        for i, face in enumerate(todo):
            try:
                face.features = font_features(face) or {}
            except Exception:  # noqa: BLE001
                face.features = {}
            if i % 50 == 49:
                self.save_cache()
        if todo:
            self.save_cache()
        self.latin_ready.set()

    def _load_cache(self) -> dict:
        try:
            data = json.loads(self.cache_path.read_text("utf-8"))
            return data["files"] if data.get("version") == CACHE_VERSION else {}
        except (OSError, ValueError, KeyError):
            return {}

    def save_cache(self) -> None:
        with self._lock:
            files: dict[str, dict] = {}
            for f in self._faces.values():
                entry = files.setdefault(f.path, {"mtime": self._mtimes.get(f.path, 0), "faces": []})
                entry["faces"].append(asdict(f))
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"version": CACHE_VERSION, "files": files}, ensure_ascii=False), "utf-8")
        tmp.replace(self.cache_path)

    def scan(self) -> None:
        cache = self._load_cache()
        faces: dict[str, FaceInfo] = {}
        sources = [(config.FONTS_DIR, True)] + [(d, False) for d in system_font_dirs()]
        seen = set()
        for directory, user in sources:
            if not directory.is_dir():
                continue
            for path in _font_files(directory):
                key = str(path)
                if key in seen:
                    continue
                seen.add(key)
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    continue
                self._mtimes[key] = mtime
                cached = cache.get(key)
                if cached and cached["mtime"] == mtime:
                    items = [FaceInfo(**f) for f in cached["faces"]]
                    for f in items:
                        f.user = user
                else:
                    items = _inspect(path, user)
                for f in items:
                    faces[f.id] = f
        with self._lock:
            self._faces = faces
        self.save_cache()
        self.ready.set()

    def get(self, face_id: str) -> FaceInfo | None:
        return self._faces.get(face_id)

    def content_fonts(self) -> list[FaceInfo]:
        """可當內容字型的中文字型，排序：開心宋體 → 使用者字型 → 宋／明體 → 涵蓋率。"""
        items = [f for f in self._faces.values()
                 if f.cjk_coverage >= 0.5 and not EXCLUDED_CONTENT.search(f.family + " " + Path(f.path).name)]
        return sorted(items, key=lambda f: (
            not PREFERRED_CONTENT.search(f.family + f.path),
            not f.user,
            not SONG_PATTERN.search(f.family + " " + f.path),
            -f.cjk_coverage,
            f.family,
        ))

    def latin_fonts(self) -> list[FaceInfo]:
        return [f for f in self._faces.values() if f.ascii]

    def content_chain(self, primary_id: str) -> list[FaceInfo]:
        """主內容字型＋其他中文字型當備援（缺字時依序遞補）。"""
        primary = self.get(primary_id)
        others = [f for f in self.content_fonts() if f.id != primary_id]
        return ([primary] if primary else []) + others[:6]
