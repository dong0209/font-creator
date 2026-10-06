"""專案與背景生成工作。

每個專案一個資料夾（work/<id>/）：
  source.png            上傳的圖片
  project.json          切字結果、特徵分析、生成設定與進度
  refs/<n>.png          風格參考圖（96px）
  previews/<run>/...    試生成結果
  glyphs/<run>/<cp>.png 正式生成的字圖，已存在的會跳過，因此中斷後可以續跑
  output/<name>.ttf     輸出字型

生成很耗時（13,000 字在 M4 上約需數小時），所以全部由單一背景執行緒依序處理，
模型只載入一次；伺服器重啟後，未完成的工作會標記為「已暫停」，可在網頁按「繼續」。
"""

import hashlib
import json
import queue
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Callable

from PIL import Image

from . import analyze, charsets, config, fontbuild, segment
from .content import ContentFont

PREVIEW_TEXT = "永東國酬愛鬱靈鷹袋Ag"


def _now() -> float:
    return round(time.time(), 1)


class Project:
    def __init__(self, root: Path):
        self.root = root
        self.lock = threading.RLock()
        self.data = json.loads((root / "project.json").read_text("utf-8"))

    @property
    def id(self) -> str:
        return self.data["id"]

    def save(self) -> None:
        with self.lock:
            tmp = self.root / "project.json.tmp"
            tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), "utf-8")
            tmp.replace(self.root / "project.json")

    def update_run(self, **fields) -> None:
        with self.lock:
            self.data.setdefault("run", {}).update(fields)
            self.save()

    def ref_image(self, index: int) -> Image.Image:
        return Image.open(self.root / "refs" / f"{index}.png").convert("RGB")


def create_project(work_dir: Path, image: Image.Image, filename: str = "") -> Project:
    pid = uuid.uuid4().hex[:12]
    root = work_dir / pid
    (root / "refs").mkdir(parents=True)
    image.save(root / "source.png")

    seg = segment.segment(image)
    report = analyze.analyze(seg)
    glyphs = []
    for g in seg.glyphs:
        segment.style_image(g).save(root / "refs" / f"{g.index}.png")
        glyphs.append({"index": g.index, "line": g.line, "score": g.score,
                       "box": [g.box.x0, g.box.y0, g.box.x1, g.box.y1]})
    ranked = sorted(glyphs, key=lambda g: -g["score"])
    data = {
        "id": pid,
        "created": _now(),
        "filename": filename,
        "analysis": report,
        "glyphs": glyphs,
        "default_ref": ranked[0]["index"] if ranked else None,
        "run": None,
        "previews": [],
    }
    (root / "project.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), "utf-8")
    return Project(root)


def run_key(settings: dict) -> str:
    """同樣的參考字、內容字型、步數、種子會得到同樣的字圖，可以共用快取。"""
    keys = ("ref", "content_font", "steps", "seed")
    raw = json.dumps({k: settings[k] for k in keys}, sort_keys=True)
    return hashlib.sha1(raw.encode()).hexdigest()[:10]


class Manager:
    def __init__(self, work_dir: Path | None = None, generator_factory: Callable | None = None):
        self.work_dir = Path(work_dir or config.WORK_DIR)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._factory = generator_factory or self._default_factory
        self._generator = None
        self._generator_error: str | None = None
        self._projects: dict[str, Project] = {}
        self._queue: queue.Queue = queue.Queue()
        self._cancel: set[str] = set()
        self._lock = threading.Lock()
        self._load_existing()
        threading.Thread(target=self._worker, daemon=True, name="fontmaker-worker").start()

    # ---------- 專案 ----------

    def _load_existing(self) -> None:
        for path in sorted(self.work_dir.glob("*/project.json")):
            try:
                p = Project(path.parent)
            except (OSError, json.JSONDecodeError):
                continue
            run = p.data.get("run")
            if run and run.get("status") in ("queued", "running"):
                p.update_run(status="paused", message="伺服器重新啟動，按「繼續」接著生成")
            self._projects[p.id] = p

    def create(self, image: Image.Image, filename: str = "") -> Project:
        p = create_project(self.work_dir, image, filename)
        self._projects[p.id] = p
        return p

    def get(self, pid: str) -> Project | None:
        return self._projects.get(pid)

    def all(self) -> list[Project]:
        return sorted(self._projects.values(), key=lambda p: -p.data["created"])

    # ---------- 生成器 ----------

    @staticmethod
    def _default_factory(steps: int):
        from .generator import FontDiffuserGenerator
        return FontDiffuserGenerator(steps=steps)

    def _get_generator(self, steps: int):
        if self._generator is None:
            self._generator = self._factory(steps)
        self._generator.steps = steps
        return self._generator

    # ---------- 工作佇列 ----------

    def submit_preview(self, p: Project, settings: dict) -> str:
        preview_id = uuid.uuid4().hex[:8]
        entry = {"id": preview_id, "status": "queued", "settings": settings, "chars": [], "created": _now()}
        with p.lock:
            p.data["previews"] = ([entry] + p.data.get("previews", []))[:6]
            p.save()
        self._queue.put(("preview", p.id, preview_id))
        return preview_id

    def submit_run(self, p: Project, settings: dict) -> None:
        run = p.data.get("run")
        if run and run.get("status") in ("queued", "running"):
            raise RuntimeError("這個專案已經在生成中")
        chars = charsets.build_charset(settings["preset"], settings.get("custom", ""), settings.get("latin", True))
        if not chars:
            raise ValueError("字表是空的")
        p.data["run"] = {**settings, "key": run_key(settings), "status": "queued", "total": len(chars),
                         "done": 0, "skipped": [], "message": "排隊中", "started": None,
                         "rate": None, "output": None}
        p.save()
        self._cancel.discard(p.id)
        self._queue.put(("run", p.id, None))

    def cancel(self, p: Project) -> None:
        run = p.data.get("run")
        if run and run.get("status") in ("queued", "running"):
            self._cancel.add(p.id)
            if run["status"] == "queued":
                p.update_run(status="paused", message="已暫停")

    def resume(self, p: Project) -> None:
        run = p.data.get("run")
        if not run or run.get("status") not in ("paused", "error"):
            raise RuntimeError("沒有可以繼續的工作")
        self._cancel.discard(p.id)
        p.update_run(status="queued", message="排隊中")
        self._queue.put(("run", p.id, None))

    def _worker(self) -> None:
        while True:
            kind, pid, extra = self._queue.get()
            p = self._projects.get(pid)
            if p is None:
                continue
            try:
                if kind == "preview":
                    self._do_preview(p, extra)
                elif kind == "run":
                    if p.data.get("run", {}).get("status") == "queued":
                        self._do_run(p)
            except Exception as exc:  # noqa: BLE001 - 任何錯誤都回報到網頁上
                traceback.print_exc()
                msg = f"{type(exc).__name__}: {exc}"
                if kind == "preview":
                    self._set_preview(p, extra, status="error", message=msg)
                else:
                    p.update_run(status="error", message=msg)

    # ---------- 試生成 ----------

    def _set_preview(self, p: Project, preview_id: str, **fields) -> None:
        with p.lock:
            for entry in p.data.get("previews", []):
                if entry["id"] == preview_id:
                    entry.update(fields)
            p.save()

    def _do_preview(self, p: Project, preview_id: str) -> None:
        entry = next((e for e in p.data.get("previews", []) if e["id"] == preview_id), None)
        if entry is None:
            return
        s = entry["settings"]
        self._set_preview(p, preview_id, status="running", message="載入模型與生成中…")
        font = ContentFont(config.FONTS_DIR / s["content_font"])
        text = [c for c in dict.fromkeys(s.get("text") or PREVIEW_TEXT) if not c.isspace()]
        items = [(c, font.render(c)) for c in text]
        items = [(c, r) for c, r in items if r is not None]
        out_dir = p.root / "previews" / preview_id
        out_dir.mkdir(parents=True, exist_ok=True)
        gen = self._get_generator(s["steps"])
        t0 = time.time()
        images = gen.generate([r[0] for _, r in items], p.ref_image(s["ref"]), seed=s["seed"])
        seconds = time.time() - t0
        sources = []
        for (c, (_, placement)), img in zip(items, images):
            img.save(out_dir / f"{ord(c):X}.png")
            sources.append(fontbuild.GlyphSource(c, img, placement))
        spec = fontbuild.FontSpec(family=f"Preview {preview_id}", glyphs=sources,
                                  weight=p.data["analysis"].get("weight", 400),
                                  space_advance=font.space_advance())
        fontbuild.build_font(spec, out_dir / "preview.ttf")
        self._set_preview(p, preview_id, status="done", chars=[c for c, _ in items],
                          seconds=round(seconds, 2), message="")

    # ---------- 正式生成 ----------

    def _do_run(self, p: Project) -> None:
        run = p.data["run"]
        font = ContentFont(config.FONTS_DIR / run["content_font"])
        chars = charsets.build_charset(run["preset"], run.get("custom", ""), run.get("latin", True))
        glyph_dir = p.root / "glyphs" / run["key"]
        glyph_dir.mkdir(parents=True, exist_ok=True)
        style = p.ref_image(run["ref"])

        todo, skipped = [], []
        for c in chars:
            if not font.has(c):
                skipped.append(c)
            elif not (glyph_dir / f"{ord(c):X}.png").exists():
                todo.append(c)
        done = len(chars) - len(skipped) - len(todo)
        p.update_run(status="running", message="載入模型中…", skipped=skipped,
                     total=len(chars) - len(skipped), done=done, started=run.get("started") or _now())

        gen = self._get_generator(run["steps"])
        t0, made = time.time(), 0
        for i in range(0, len(todo), gen.batch_size):
            if p.id in self._cancel:
                self._cancel.discard(p.id)
                p.update_run(status="paused", message="已暫停，按「繼續」接著生成")
                return
            batch = []
            for c in todo[i : i + gen.batch_size]:
                r = font.render(c)
                if r is None:
                    skipped.append(c)
                else:
                    batch.append((c, r[0]))
            images = gen.generate([img for _, img in batch], style, seed=run["seed"] + i)
            for (c, _), img in zip(batch, images):
                img.save(glyph_dir / f"{ord(c):X}.png")
            made += len(batch)
            done += len(batch)
            rate = (time.time() - t0) / made
            p.update_run(done=done, rate=round(rate, 3), skipped=skipped,
                         message=f"生成中（每字約 {rate:.2f} 秒）")

        p.update_run(message="組合字型檔中…")
        self._build(p, font, chars, glyph_dir)

    def _build(self, p: Project, font: ContentFont, chars: list[str], glyph_dir: Path) -> None:
        run = p.data["run"]
        sources = []
        for c in chars:
            path = glyph_dir / f"{ord(c):X}.png"
            r = font.render(c) if path.exists() else None
            if r is not None:
                sources.append(fontbuild.GlyphSource(c, Image.open(path).convert("L"), r[1]))
        family = fontbuild.nc_family_name(run["family"])
        out = p.root / "output" / f"{fontbuild._ps_name(family)}.ttf"
        analysis = p.data["analysis"]
        spec = fontbuild.FontSpec(family=family, glyphs=sources, weight=analysis.get("weight", 400),
                                  italic_angle=analysis.get("slant", 0.0), space_advance=font.space_advance())
        info = fontbuild.build_font(spec, out)
        p.update_run(status="done", message="完成", output=out.name, family=info["family"],
                     glyph_count=info["glyphs"], empty=info["empty"], finished=_now())
