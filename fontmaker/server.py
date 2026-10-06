"""網頁介面與 API。執行：python -m fontmaker.server（預設 http://127.0.0.1:8765）"""

import io
import os
import re

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from . import charsets, config, generator, jobs

MAX_UPLOAD = 30 * 1024 * 1024


class PreviewRequest(BaseModel):
    ref: int
    content_font: str
    steps: int = Field(20, ge=5, le=50)
    seed: int = 123
    text: str = Field(jobs.PREVIEW_TEXT, max_length=40)


class RunRequest(BaseModel):
    family: str = Field(..., min_length=1, max_length=40)
    preset: str = "common"
    custom: str = Field("", max_length=20000)
    latin: bool = True
    ref: int
    content_font: str
    steps: int = Field(20, ge=5, le=50)
    seed: int = 123


def create_app(manager: jobs.Manager | None = None, require_setup: bool = True) -> FastAPI:
    manager = manager or jobs.Manager()
    app = FastAPI(title="NC Font Maker")
    web_dir = config.ROOT / "web"

    def project_or_404(pid: str) -> jobs.Project:
        p = manager.get(pid)
        if p is None:
            raise HTTPException(404, "找不到這個專案")
        return p

    def check_inputs(p: jobs.Project, ref: int, content_font: str) -> None:
        if require_setup:
            problems = generator.check_setup()
            if problems:
                raise HTTPException(409, "\n".join(problems))
        if content_font not in {f.name for f in config.list_content_fonts()}:
            raise HTTPException(400, "請先選擇內容字型")
        if not (p.root / "refs" / f"{ref}.png").is_file():
            raise HTTPException(400, "參考字不存在")

    @app.get("/")
    def index():
        return FileResponse(web_dir / "index.html")

    app.mount("/static", StaticFiles(directory=web_dir), name="static")

    @app.get("/api/status")
    def status():
        return {
            "problems": generator.check_setup() if require_setup else [],
            "content_fonts": [f.name for f in config.list_content_fonts()],
            "fonts_dir": str(config.FONTS_DIR),
            "presets": [{"id": k, "label": v} for k, v in charsets.PRESETS.items()],
            "preview_text": jobs.PREVIEW_TEXT,
        }

    @app.post("/api/projects")
    async def upload(file: UploadFile = File(...)):
        raw = await file.read(MAX_UPLOAD + 1)
        if len(raw) > MAX_UPLOAD:
            raise HTTPException(413, "圖片超過 30MB")
        try:
            image = Image.open(io.BytesIO(raw))
            image.load()
        except (UnidentifiedImageError, OSError):
            raise HTTPException(400, "無法讀取這個圖片檔")
        # 切字與分析很吃 CPU，放到執行緒池，避免卡住其他請求
        p = await run_in_threadpool(manager.create, image, file.filename or "")
        return p.data

    @app.get("/api/projects")
    def list_projects():
        return [{"id": p.id, "created": p.data["created"], "filename": p.data.get("filename", ""),
                 "glyphs": len(p.data["glyphs"]), "run": p.data.get("run")} for p in manager.all()]

    @app.get("/api/projects/{pid}")
    def get_project(pid: str):
        return project_or_404(pid).data

    @app.get("/api/projects/{pid}/source")
    def source(pid: str):
        return FileResponse(project_or_404(pid).root / "source.png")

    @app.get("/api/projects/{pid}/refs/{index}.png")
    def ref(pid: str, index: int):
        path = project_or_404(pid).root / "refs" / f"{index}.png"
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path)

    @app.post("/api/projects/{pid}/preview")
    def preview(pid: str, req: PreviewRequest):
        p = project_or_404(pid)
        check_inputs(p, req.ref, req.content_font)
        return {"id": manager.submit_preview(p, req.model_dump())}

    @app.get("/api/projects/{pid}/previews/{preview_id}/{name}")
    def preview_file(pid: str, preview_id: str, name: str):
        if not re.fullmatch(r"[0-9a-f]{8}", preview_id) or not re.fullmatch(r"([0-9A-F]+\.png|preview\.ttf)", name):
            raise HTTPException(404)
        path = project_or_404(pid).root / "previews" / preview_id / name
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path)

    @app.post("/api/projects/{pid}/run")
    def run(pid: str, req: RunRequest):
        p = project_or_404(pid)
        check_inputs(p, req.ref, req.content_font)
        if req.preset not in charsets.PRESETS:
            raise HTTPException(400, "未知的字集")
        try:
            manager.submit_run(p, req.model_dump())
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(409, str(exc))
        return p.data["run"]

    @app.post("/api/projects/{pid}/cancel")
    def cancel(pid: str):
        p = project_or_404(pid)
        manager.cancel(p)
        return p.data.get("run")

    @app.post("/api/projects/{pid}/resume")
    def resume(pid: str):
        p = project_or_404(pid)
        try:
            manager.resume(p)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        return p.data["run"]

    @app.get("/api/projects/{pid}/font")
    def download(pid: str, inline: bool = False):
        p = project_or_404(pid)
        run = p.data.get("run") or {}
        if run.get("status") != "done" or not run.get("output"):
            raise HTTPException(404, "字型尚未完成")
        path = p.root / "output" / run["output"]
        if inline:
            return FileResponse(path, media_type="font/ttf")
        return FileResponse(path, media_type="font/ttf", filename=run["output"])

    @app.get("/favicon.ico")
    def favicon():
        return RedirectResponse("/static/favicon.svg")

    return app


def main() -> None:
    import uvicorn

    host = os.environ.get("FONTMAKER_HOST", "127.0.0.1")
    port = int(os.environ.get("FONTMAKER_PORT", "8765"))
    print(f"NC Font Maker：http://{host}:{port}")
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()
