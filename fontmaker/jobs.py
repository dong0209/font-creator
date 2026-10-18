"""專案與背景生成工作。

每個專案一個資料夾（work/<id>/）：
  source.png            上傳的圖片
  project.json          切字結果、特徵分析、生成設定與進度
  refs/<n>.png          風格參考圖（96px）
  previews/<id>/...     試生成結果
  glyphs/<run>/<cp>.png 正式生成的字圖（多參考字時為選出的最佳結果），已存在的會跳過，中斷後可續跑
  output/<name>.ttf     輸出字型

中文字與標點由 FontDiffuser 生成；英數字取自風格最接近的現成字型（latin.py）。
生成很耗時，全部由單一背景執行緒依序處理，模型只載入一次；
伺服器重啟後，未完成的工作會標記為「已暫停」，可在網頁按「繼續」。
"""

import hashlib
import json
import re
import shutil
import zipfile
import queue
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Callable

from PIL import Image

from . import analyze, charsets, config, fontbuild, latin, quality, segment, specimen, vectorize
from .content import ContentChain
from .fontscan import FontScanner

PREVIEW_TEXT = "永東國酬愛鬱靈鷹袋Ag"
MAX_REFS = 3


def _now() -> float:
    return round(time.time(), 1)


class Project:
    def __init__(self, root: Path):
        self.root = root
        self.lock = threading.RLock()
        self.data = json.loads((root / "project.json").read_text("utf-8"))
        if "groups" not in self.data:  # 舊版專案：整張圖當作一組
            self.data["groups"] = [{"id": "g1", "name": "字型 A", "glyphs": [g["index"] for g in self.data["glyphs"]],
                                    "analysis": self.data["analysis"], "default_ref": self.data.get("default_ref")}]

    def group(self, group_id: str | None) -> dict:
        groups = self.data["groups"]
        return next((g for g in groups if g["id"] == group_id), groups[0]) if groups else {"analysis": self.data["analysis"]}

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


def normalize_image(image: Image.Image) -> Image.Image:
    """統一成 RGB／RGBA：處理 EXIF 旋轉、CMYK、16-bit、調色盤、動畫第一格等格式。"""
    from PIL import ImageOps

    if getattr(image, "is_animated", False):
        image.seek(0)
    image = ImageOps.exif_transpose(image)
    if image.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        import numpy as np

        a = np.asarray(image, dtype=np.float64)
        hi = a.max() or 1
        image = Image.fromarray((a / hi * 255).astype("uint8"))
    has_alpha = image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info
    return image.convert("RGBA" if has_alpha else "RGB")


def _group_entry(seg, number: int, ids: list[int]) -> dict:
    report = analyze.analyze(seg, ids)
    ranked = sorted(ids, key=lambda i: -seg.glyphs[i].score)
    return {"id": f"g{number}", "name": f"字型 {chr(ord('A') + number - 1) if number <= 26 else number}",
            "glyphs": ids, "analysis": report, "default_ref": ranked[0] if ranked else None}


def create_project(work_dir: Path, image: Image.Image, filename: str = "") -> Project:
    pid = uuid.uuid4().hex[:12]
    root = work_dir / pid
    (root / "refs").mkdir(parents=True)
    image = normalize_image(image)
    image.save(root / "source.png")

    seg = segment.segment(image)
    report = analyze.analyze(seg)
    glyphs = []
    for g in seg.glyphs:
        segment.style_image(g).save(root / "refs" / f"{g.index}.png")
        glyphs.append({"index": g.index, "line": g.line, "score": g.score,
                       "box": [g.box.x0, g.box.y0, g.box.x1, g.box.y1]})
    groups = [_group_entry(seg, n, ids) for n, ids in enumerate(analyze.group_glyphs(seg), start=1)]
    for group in groups:
        for i in group["glyphs"]:
            glyphs[i]["group"] = group["id"]
    ranked = sorted(glyphs, key=lambda g: -g["score"])
    data = {
        "id": pid,
        "created": _now(),
        "filename": filename,
        "analysis": report,
        "glyphs": glyphs,
        "groups": groups,
        "default_ref": ranked[0]["index"] if ranked else None,
        "run": None,
        "previews": [],
    }
    (root / "project.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), "utf-8")
    return Project(root)


def run_key(settings: dict) -> str:
    """同樣的參考字組合、內容字型、步數、種子會得到同樣的字圖，可以共用快取。"""
    raw = json.dumps({"refs": sorted(settings["refs"]), "content_font": settings["content_font"],
                      "steps": settings["steps"], "seed": settings["seed"]}, sort_keys=True)
    return hashlib.sha1(raw.encode()).hexdigest()[:10]


def split_chars(chars: list[str]) -> tuple[list[str], list[str]]:
    """分成 (AI 生成的中文與標點, 取自現成字型的英數字)。"""
    ai = [c for c in chars if c not in latin.LATIN_CHARS]
    return ai, [c for c in chars if c in latin.LATIN_CHARS]


def _safe_dirname(name: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "", name).strip(" .")
    return cleaned or "font"


class Manager:
    def __init__(self, work_dir: Path | None = None, generator_factory: Callable | None = None,
                 scanner: FontScanner | None = None):
        self.work_dir = Path(work_dir or config.WORK_DIR)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.scanner = scanner or FontScanner(self.work_dir / "fontscan.json")
        if not self.scanner.ready.is_set():
            self.scanner.start()
        self._factory = generator_factory or self._default_factory
        self._generator = None
        self._projects: dict[str, Project] = {}
        self._queue: queue.Queue = queue.Queue()
        self._cancel: set[str] = set()
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
            stale = [v for v in p.data.get("previews", []) if v.get("status") in ("queued", "running")]
            for v in stale:
                v.update(status="error", message="伺服器重新啟動，這次試生成已中斷，請重新按「試生成」")
            if stale:
                p.save()
            self._projects[p.id] = p

    def create(self, image: Image.Image, filename: str = "") -> Project:
        p = create_project(self.work_dir, image, filename)
        self._projects[p.id] = p
        return p

    def get(self, pid: str) -> Project | None:
        return self._projects.get(pid)

    def all(self) -> list[Project]:
        return sorted(self._projects.values(), key=lambda p: -p.data["created"])

    # ---------- 字型 ----------

    def content_chain(self, face_id: str) -> ContentChain:
        self.scanner.ready.wait()
        return ContentChain.from_faces(self.scanner.content_chain(face_id))

    def latin_candidates(self, analysis: dict, limit: int = 8) -> list[tuple]:
        target = latin.target_from_analysis(analysis)
        return latin.rank(target, self.scanner.latin_fonts(), limit)

    def latin_face(self, analysis: dict, face_id: str):
        self.scanner.ready.wait()
        face = self.scanner.get(face_id) if face_id else None
        if face is None:
            self.scanner.latin_ready.wait()
            ranked = self.latin_candidates(analysis, 1)
            face = ranked[0][0] if ranked else None
        return face

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

    def _generate_best(self, gen, p: Project, refs: list[int], contents: list[Image.Image], seed: int) -> list:
        """每個參考字各生成一次，逐字挑出最好的結果。"""
        candidates = [gen.generate(contents, p.ref_image(r), seed=seed + i * 1000) for i, r in enumerate(refs)]
        best = []
        for j, content in enumerate(contents):
            options = [c[j] for c in candidates]
            idx, _ = quality.pick_best(options, content)
            best.append(options[idx])
        return best

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
        """settings：共用設定（字集、內容字型、步數、種子）＋ fonts：[{group, family, refs, latin_font}]。"""
        run = p.data.get("run")
        if run and run.get("status") in ("queued", "running"):
            raise RuntimeError("這個專案已經在生成中")
        chars = charsets.build_charset(settings["preset"], settings.get("custom", ""), settings.get("latin", True))
        if not chars:
            raise ValueError("字表是空的")
        if not settings.get("fonts"):
            raise ValueError("請至少選擇一組要匯出的字型")
        families = [fontbuild.nc_family_name(f["family"]) for f in settings["fonts"]]
        if len(set(families)) != len(families):
            raise ValueError("各組字型的名稱不能重複")
        ai_chars, _ = split_chars(chars)
        shared = {k: v for k, v in settings.items() if k != "fonts"}
        fonts = []
        for f, family in zip(settings["fonts"], families):
            entry = {"group": f["group"], "family": family, "refs": f["refs"], "latin_font": f.get("latin_font", "")}
            entry["key"] = run_key({**shared, "refs": f["refs"]})
            entry.update(status="pending", done=0, total=len(ai_chars), skipped=[], outputs=[])
            fonts.append(entry)
        p.data["run"] = {**shared, "fonts": fonts, "status": "queued", "total": len(ai_chars) * len(fonts),
                         "done": 0, "message": "排隊中", "started": None, "rate": None, "package": None}
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

    # ---------- 共用：組字型 ----------

    def _prepare(self, chain: ContentChain, images: dict[str, Image.Image]) -> list[dict]:
        """量每個 AI 字的筆畫粗細，算出一致化修正。回傳 [{char, image, placement, delta, stroke}]。"""
        items = []
        gen_w, content_w = {}, {}
        for c, img in images.items():
            r = chain.render(c)
            if r is None:
                continue
            g = quality.binarize(img)
            gen_w[c] = vectorize.stroke_width(g) if g.any() else 0.0
            cb = quality.content_binary(r[0], img.size[0])
            content_w[c] = vectorize.stroke_width(cb) if cb.any() else 0.0
            items.append({"char": c, "image": img, "placement": r[1]})
        corrections = quality.weight_corrections(gen_w, content_w)
        for it in items:
            it["delta"] = corrections.get(it["char"], 0.0)
            it["stroke"] = gen_w[it["char"]] + 2 * it["delta"]  # 一致化後的粗細
        return items

    def _sources(self, items: list[dict], latin_chars: list[str], face, analysis: dict,
                 ratio: float = 1.0) -> tuple[list, int | None]:
        """組成一個字重的 GlyphSource 清單。ratio：相對於分析字重的筆畫粗細倍率。"""
        sources = [fontbuild.GlyphSource(it["char"], it["image"], it["placement"],
                                         weight_delta=it["delta"] + (ratio - 1) * it["stroke"] / 2)
                   for it in items]
        space = None
        if latin_chars and face is not None:
            target = latin.target_from_analysis(analysis)
            target["stroke"] *= ratio
            styler = latin.LatinStyler(face, target)
            space = styler.space_advance()
            for c in latin_chars:
                g = styler.glyph(c)
                if g is not None:
                    sources.append(fontbuild.GlyphSource(c, contours=g[0], advance=g[1]))
        return sources, space

    @staticmethod
    def _spec(analysis: dict, family: str, sources: list, chain: ContentChain, space: int | None,
              weight: int | None = None, style: str = "Regular"):
        return fontbuild.FontSpec(family=family, glyphs=sources, weight=weight or analysis.get("weight", 400),
                                  style=style, italic_angle=analysis.get("slant", 0.0),
                                  space_advance=(space / 1000) if space else chain.space_advance())

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
        analysis = p.group(s.get("group"))["analysis"]
        self._set_preview(p, preview_id, status="running", message="載入模型與生成中…")
        chain = self.content_chain(s["content_font"])
        text = [c for c in dict.fromkeys(s.get("text") or PREVIEW_TEXT) if not c.isspace()]
        ai_chars, latin_chars = split_chars(text)
        items = [(c, chain.render(c)) for c in ai_chars]
        items = [(c, r) for c, r in items if r is not None]
        out_dir = p.root / "previews" / preview_id
        out_dir.mkdir(parents=True, exist_ok=True)

        images, seconds = {}, 0.0
        if items:
            gen = self._get_generator(s["steps"])
            t0 = time.time()
            results = self._generate_best(gen, p, s["refs"], [r[0] for _, r in items], s["seed"])
            seconds = time.time() - t0
            for (c, _), img in zip(items, results):
                img.save(out_dir / f"{ord(c):X}.png")
                images[c] = img
        face = self.latin_face(analysis, s.get("latin_font", "")) if latin_chars else None
        sources, space = self._sources(self._prepare(chain, images), latin_chars, face, analysis)
        fontbuild.build_font(self._spec(analysis, f"Preview {preview_id}", sources, chain, space), out_dir / "preview.ttf")
        self._set_preview(p, preview_id, status="done", chars=[c for c, _ in items], latin=latin_chars,
                          latin_font=face.label if face else None, seconds=round(seconds, 2),
                          per_glyph=round(seconds / len(items), 3) if items else None, message="")

    # ---------- 正式生成 ----------

    def _update_font(self, p: Project, index: int, **fields) -> None:
        with p.lock:
            p.data["run"]["fonts"][index].update(fields)
            p.save()

    def _do_run(self, p: Project) -> None:
        run = p.data["run"]
        chain = self.content_chain(run["content_font"])
        chars = charsets.build_charset(run["preset"], run.get("custom", ""), run.get("latin", True))
        ai_chars, latin_chars = split_chars(chars)
        p.update_run(status="running", message="準備中…", started=run.get("started") or _now())

        # 1. 依序生成每一組的字圖（已生成的會跳過，可中斷續跑）
        for index, font in enumerate(run["fonts"]):
            if not self._generate_font(p, index, chain, ai_chars):
                return  # 已暫停

        # 2. 描圖、衍生字重、輸出各種格式與樣張
        out_root = p.root / "output"
        if out_root.exists():
            shutil.rmtree(out_root)
        for index, font in enumerate(run["fonts"]):
            self._build_family(p, index, chain, ai_chars, latin_chars, out_root)

        # 3. 打包
        p.update_run(message="打包字型包中…")
        package = self._package(p, out_root)
        p.update_run(status="done", message="完成", package=package.name, finished=_now())

    def _generate_font(self, p: Project, index: int, chain: ContentChain, ai_chars: list[str]) -> bool:
        run = p.data["run"]
        font = run["fonts"][index]
        glyph_dir = p.root / "glyphs" / font["key"]
        glyph_dir.mkdir(parents=True, exist_ok=True)
        todo, skipped = [], []
        for c in ai_chars:
            if not chain.has(c):
                skipped.append(c)
            elif not (glyph_dir / f"{ord(c):X}.png").exists():
                todo.append(c)
        total = len(ai_chars) - len(skipped)
        self._update_font(p, index, status="generating", skipped=skipped, total=total, done=total - len(todo))
        self._sync_totals(p)

        gen = self._get_generator(run["steps"]) if todo else None
        t0, made = time.time(), 0
        step = gen.batch_size if gen else 1
        label = f"{font['family']}（第 {index + 1} / {len(run['fonts'])} 套）"
        for i in range(0, len(todo), step):
            if p.id in self._cancel:
                self._cancel.discard(p.id)
                p.update_run(status="paused", message="已暫停，按「繼續」接著生成")
                return False
            batch = [(c, chain.render(c)) for c in todo[i : i + step]]
            batch = [(c, r[0]) for c, r in batch if r is not None]
            images = self._generate_best(gen, p, font["refs"], [img for _, img in batch], run["seed"] + i)
            for (c, _), img in zip(batch, images):
                img.save(glyph_dir / f"{ord(c):X}.png")
            made += len(batch)
            rate = (time.time() - t0) / made
            self._update_font(p, index, done=total - len(todo) + i + len(batch))
            note = getattr(gen, "fallback_reason", None)
            self._sync_totals(p, rate=round(rate, 3),
                              message=f"生成 {label}中（每字約 {rate:.2f} 秒）" + (f"\n{note}" if note else ""))
        self._update_font(p, index, status="generated")
        return True

    def _sync_totals(self, p: Project, **fields) -> None:
        fonts = p.data["run"]["fonts"]
        p.update_run(done=sum(f["done"] for f in fonts), total=sum(f["total"] for f in fonts), **fields)

    def _build_family(self, p: Project, index: int, chain: ContentChain, ai_chars: list[str],
                      latin_chars: list[str], out_root: Path) -> None:
        run = p.data["run"]
        font = run["fonts"][index]
        analysis = p.group(font["group"])["analysis"]
        glyph_dir = p.root / "glyphs" / font["key"]
        label = f"{font['family']}（第 {index + 1} / {len(run['fonts'])} 套）"
        p.update_run(message=f"{label}：分析筆畫粗細…")
        self._update_font(p, index, status="building")
        images = {}
        for c in ai_chars:
            path = glyph_dir / f"{ord(c):X}.png"
            if path.exists():
                images[c] = Image.open(path).convert("L")
        items = self._prepare(chain, images)
        face = self.latin_face(analysis, font.get("latin_font", "")) if latin_chars else None

        family = font["family"]
        folder = out_root / _safe_dirname(family)
        base_weight = analysis.get("weight", 400)
        variants = analyze.weight_variants(base_weight)
        base_stroke = analyze.stroke_for_weight(base_weight)
        outputs = []
        for vi, (style, weight) in enumerate(variants):
            ratio = analyze.stroke_for_weight(weight) / base_stroke
            sources, space = self._sources(items, latin_chars, face, analysis, ratio)
            stem = f"{fontbuild._ps_name(family)}-{style.replace(' ', '')}"
            msg = f"{label}：描圖 {style}（{vi + 1} / {len(variants)} 種字重）"
            info = fontbuild.build_font(
                self._spec(analysis, family, sources, chain, space, weight, style), folder / "TTF" / f"{stem}.ttf",
                progress=lambda i, n, msg=msg: p.update_run(message=f"{msg}…（{i:,} / {n:,}）"),
                formats=("ttf", "otf", "woff2"))
            files = info["files"]
            # 依格式分資料夾
            for fmt, folder_name in (("otf", "OTF"), ("woff2", "WOFF2")):
                dest = folder / folder_name / Path(files[fmt]).name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(files[fmt], dest)
                files[fmt] = str(dest)
            outputs.append({"style": style, "weight": weight, "glyphs": info["glyphs"], "empty": info["empty"],
                            "files": {k: str(Path(v).relative_to(out_root)) for k, v in files.items()}})

        p.update_run(message=f"{label}：產生樣張…")
        self._write_css(folder, family, outputs, out_root)
        weights = [{"style": o["style"], "weight": o["weight"], "ttf": str(out_root / o["files"]["ttf"])} for o in outputs]
        ui_font = str(chain.fonts[0].path)
        specimen.render_specimen(family, weights, folder / "樣張.png", ui_font)
        regular = min(weights, key=lambda w: abs(w["weight"] - base_weight))
        specimen.render_glyph_chart(family, regular["ttf"], folder / "全字表.pdf", ui_font)
        self._update_font(p, index, status="done", outputs=outputs, folder=folder.name,
                          latin_font_name=face.label if face else None,
                          glyph_count=outputs[0]["glyphs"] if outputs else 0)

    @staticmethod
    def _write_css(folder: Path, family: str, outputs: list[dict], out_root: Path) -> None:
        rules = []
        for o in outputs:
            woff2 = Path(o["files"]["woff2"]).relative_to(folder.name)
            rules.append(
                "@font-face {\n"
                f'  font-family: "{family}";\n'
                f'  src: url("{woff2.as_posix()}") format("woff2");\n'
                f"  font-weight: {o['weight']};\n"
                "  font-style: normal;\n"
                "  font-display: swap;\n"
                "}\n")
        example = (f'/* 用法：<link rel="stylesheet" href="fonts.css">，再設定\n'
                   f'   body {{ font-family: "{family}", sans-serif; }}\n'
                   "   此字型禁止用於商業用途。 */\n\n")
        (folder / "fonts.css").write_text(example + "\n".join(rules), "utf-8")

    @staticmethod
    def _package(p: Project, out_root: Path) -> Path:
        names = [f["family"] for f in p.data["run"]["fonts"]]
        stem = _safe_dirname(names[0] if len(names) == 1 else f"{names[0]} 等 {len(names)} 套字型")
        for old in p.root.glob("*.zip"):
            old.unlink()
        zip_path = p.root / f"{stem}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(out_root.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(out_root).as_posix())
        return zip_path
