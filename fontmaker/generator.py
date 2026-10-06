"""FontDiffuser 推論介面。

FontDiffuser 原始碼與權重都不隨本專案散布（其 repo 未附授權條款），
由使用者以 scripts/setup_fontdiffuser.py 下載原始碼、自行下載權重放到 vendor/ckpt。
"""

import os
import sys
import threading
from pathlib import Path

from PIL import Image

from . import config

# torchvision 的 DeformConv2d 沒有 Apple Silicon（MPS）實作，要讓它退回 CPU 執行
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

SIZE = 96
DEFAULT_BATCH = {"cuda": 32, "mps": 16, "cpu": 8}


class GeneratorUnavailable(RuntimeError):
    pass


def check_setup() -> list[str]:
    """回傳尚未完成的安裝步驟（空清單表示可以生成）。"""
    problems = []
    if not (config.FONTDIFFUSER_DIR / "src" / "model.py").is_file():
        problems.append(f"找不到 FontDiffuser 原始碼（{config.FONTDIFFUSER_DIR}），請執行 python scripts/setup_fontdiffuser.py")
    missing = config.missing_ckpt_files()
    if missing:
        problems.append(f"找不到權重檔 {', '.join(missing)}，請下載後放到 {config.CKPT_DIR}")
    try:
        import torch  # noqa: F401
    except ImportError:
        problems.append("尚未安裝 PyTorch，請執行 pip install -r requirements.txt")
    return problems


def pick_device(preference: str = "auto") -> str:
    import torch

    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class FontDiffuserGenerator:
    """載入一次模型，之後可重複批次生成。"""

    def __init__(self, ckpt_dir: Path | None = None, device: str | None = None,
                 steps: int = 20, guidance_scale: float = 7.5, random_weights: bool = False):
        problems = [p for p in check_setup() if random_weights is False or "權重" not in p]
        if problems:
            raise GeneratorUnavailable("\n".join(problems))

        fd_dir = str(config.FONTDIFFUSER_DIR)
        if fd_dir not in sys.path:
            sys.path.insert(0, fd_dir)

        import torch
        import torchvision.transforms as T
        from configs.fontdiffuser import get_parser
        from src import (FontDiffuserDPMPipeline, FontDiffuserModelDPM, build_content_encoder,
                         build_ddpm_scheduler, build_style_encoder, build_unet)

        self.torch = torch
        self.device = pick_device(device or config.DEVICE)
        self.steps = steps
        self.batch_size = config.BATCH_SIZE or DEFAULT_BATCH.get(self.device, 8)
        self._lock = threading.Lock()

        args = get_parser().parse_args([])
        args.style_image_size = (SIZE, SIZE)
        args.content_image_size = (SIZE, SIZE)
        args.guidance_scale = guidance_scale
        self.args = args

        unet, style_encoder, content_encoder = build_unet(args), build_style_encoder(args), build_content_encoder(args)
        if not random_weights:
            ckpt = Path(ckpt_dir or config.CKPT_DIR)
            for module, name in ((unet, "unet.pth"), (style_encoder, "style_encoder.pth"),
                                 (content_encoder, "content_encoder.pth")):
                module.load_state_dict(torch.load(ckpt / name, map_location="cpu"))
        model = FontDiffuserModelDPM(unet=unet, style_encoder=style_encoder, content_encoder=content_encoder)
        model.to(self.device).eval()

        self.pipe = FontDiffuserDPMPipeline(
            model=model,
            ddpm_train_scheduler=build_ddpm_scheduler(args),
            model_type=args.model_type,
            guidance_type=args.guidance_type,
            guidance_scale=args.guidance_scale,
        )
        self.transform = T.Compose([
            T.Resize((SIZE, SIZE), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])

    def generate(self, contents: list[Image.Image], style: Image.Image, seed: int = 123) -> list[Image.Image]:
        """contents：內容字圖（128×128 RGB）；style：風格參考圖。回傳 96×96 生成結果。"""
        torch = self.torch
        if not contents:
            return []
        with self._lock, torch.no_grad():
            content = torch.stack([self.transform(c.convert("RGB")) for c in contents]).to(self.device)
            style_t = self.transform(style.convert("RGB"))[None].repeat(len(contents), 1, 1, 1).to(self.device)
            generator = torch.Generator().manual_seed(seed)
            a = self.args
            return self.pipe.generate(
                content_images=content,
                style_images=style_t,
                batch_size=len(contents),
                order=a.order,
                num_inference_step=self.steps,
                content_encoder_downsample_size=a.content_encoder_downsample_size,
                t_start=a.t_start,
                t_end=a.t_end,
                dm_size=a.content_image_size,
                algorithm_type=a.algorithm_type,
                skip_type=a.skip_type,
                method=a.method,
                correcting_x0_fn=a.correcting_x0_fn,
                generator=generator,
            )
