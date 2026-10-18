"""各種輸入情境的回歸測試：圖片格式、版面、照片、字體風格判斷、命名、重啟。"""

import io
import os
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from fontmaker import analyze, fontbuild, jobs, segment
from fontmaker.server import create_app

from .conftest import FakeGenerator


def layout(font_path: Path, lines: list[str], size: int = 120, gap: float = 1.15) -> Image.Image:
    font = ImageFont.truetype(str(font_path), size)
    w = size * (max(map(len, lines)) + 2)
    h = int(size * gap * len(lines)) + size
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    for i, t in enumerate(lines):
        d.text((size // 2, size // 2 + i * int(size * gap)), t, font=font, fill=0)
    return img.convert("RGB")


# ---------- 切字 ----------

@pytest.mark.parametrize("char", list("鬱三意圭二八小心川一口"))
def test_single_character_is_one_glyph(cjk_font_path, char):
    assert len(segment.segment(layout(cjk_font_path, [char])).glyphs) == 1


@pytest.mark.parametrize("name, lines, expected", [
    ("直排", list("永和九年歲"), 5),
    ("2x2", ["永和", "九年"], 4),
    ("2x2 橫畫字", ["一二", "三四"], 4),
    ("3x3", ["永和九", "年歲在", "癸丑暮"], 9),
    ("兩行", ["永和九年歲", "在癸丑暮春"], 10),
    ("橫畫字", ["一二三四五六七八九十"], 10),
    ("標點", ["你好，世界。"], 6),
    ("左右結構", ["你好明川"], 4),
])
def test_layouts(cjk_font_path, name, lines, expected):
    assert len(segment.segment(layout(cjk_font_path, lines)).glyphs) == expected, name


def test_photo_with_uneven_lighting(cjk_font_path):
    img = np.array(layout(cjk_font_path, ["永和九年歲在癸丑", "暮春之初會於會稽"], size=80).convert("L"), np.float32)
    h, w = img.shape
    img = img * (np.linspace(0.55, 1.0, w)[None, :] * np.linspace(0.7, 1.0, h)[:, None])
    img = img + np.random.default_rng(1).normal(0, 8, img.shape)
    photo = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))
    buf = io.BytesIO()
    photo.save(buf, "JPEG", quality=55)
    seg = segment.segment(Image.open(buf))
    assert [len(line) for line in seg.lines] == [8, 8]
    assert analyze.analyze(seg)["style"] == "黑體風格"


# ---------- 特徵分析 ----------

def test_punctuation_does_not_look_handwritten(cjk_font_path):
    report = analyze.analyze(segment.segment(layout(cjk_font_path, ["永和九年，歲在癸丑。"], size=80)))
    assert report["style"] == "黑體風格"


def test_upright_text_has_zero_slant(cjk_font_path):
    report = analyze.analyze(segment.segment(layout(cjk_font_path, ["永和九年歲在癸丑"], size=80)))
    assert abs(report["slant"]) < 1


def test_latin_only_image_warns(cjk_font_path):
    report = analyze.analyze(segment.segment(layout(cjk_font_path, ["Hello World"], size=80)))
    assert "英文" in report["style"]
    assert any("英文" in w for w in report["warnings"])


def test_small_text_warns(cjk_font_path):
    report = analyze.analyze(segment.segment(layout(cjk_font_path, ["永和九年歲在癸丑"], size=24)))
    assert any("太小" in w for w in report["warnings"])


STYLE_FONTS = {  # 額外字型（放在 FONTMAKER_TEST_FONTS_DIR）→ 預期風格
    "NotoSerifTC-Regular.otf": "明體／宋體風格",
    "huninn.ttf": "圓體風格",
    "wenkai.ttf": "楷書風格",
}


@pytest.mark.parametrize("filename, expected", list(STYLE_FONTS.items()))
def test_style_classification(filename, expected):
    folder = os.environ.get("FONTMAKER_TEST_FONTS_DIR")
    path = Path(folder) / filename if folder else None
    if path is None or not path.is_file():
        pytest.skip(f"需要 {filename}（設定 FONTMAKER_TEST_FONTS_DIR）")
    report = analyze.analyze(segment.segment(layout(path, ["永和九年歲在癸丑", "暮春之初會於會稽"], size=80)))
    assert report["style"] == expected


# ---------- 上傳格式 ----------

@pytest.fixture
def client(tmp_path, fonts_dir):
    manager = jobs.Manager(work_dir=tmp_path / "work", generator_factory=lambda steps: FakeGenerator())
    manager.scanner.ready.wait(60)
    return TestClient(create_app(manager, require_setup=False), raise_server_exceptions=False), manager


def _encode(img: Image.Image, fmt: str, **kw) -> bytes:
    buf = io.BytesIO()
    img.save(buf, fmt, **kw)
    return buf.getvalue()


def test_upload_formats(client, cjk_font_path):
    c, _ = client
    base = layout(cjk_font_path, ["永和九年", "歲在癸丑"], size=80)
    rgba = Image.new("RGBA", base.size, (0, 0, 0, 0))
    rgba.putalpha(Image.eval(base.convert("L"), lambda v: 255 - v))
    gray16 = Image.fromarray(np.array(base.convert("L")).astype(np.uint16) * 257)
    cases = {
        "cmyk.jpg": _encode(base.convert("CMYK"), "JPEG"),
        "gray16.png": _encode(gray16, "PNG"),
        "alpha.png": _encode(rgba, "PNG"),
        "palette.gif": _encode(base.convert("P"), "GIF"),
        "a.webp": _encode(base, "WEBP"),
    }
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
        cases["photo.heic"] = _encode(base, "HEIF")
    except ImportError:
        pass
    for name, data in cases.items():
        res = c.post("/api/projects", files={"file": (name, data, "application/octet-stream")})
        assert res.status_code == 200, (name, res.text)
        assert len(res.json()["glyphs"]) == 8, name


def test_upload_rejects_bad_files(client):
    c, _ = client
    huge = _encode(Image.new("1", (14000, 13000), 1), "PNG")
    assert c.post("/api/projects", files={"file": ("big.png", huge, "image/png")}).status_code == 413
    assert c.post("/api/projects", files={"file": ("x.png", b"not an image", "image/png")}).status_code == 400
    assert c.post("/api/projects", files={"file": ("e.png", b"", "image/png")}).status_code == 400


# ---------- 命名與重啟 ----------

def test_chinese_family_names_get_unique_ps_names():
    a = fontbuild._ps_name(fontbuild.nc_family_name("我的手寫字"))
    b = fontbuild._ps_name(fontbuild.nc_family_name("另一套字型"))
    assert a != b and a.endswith("NC") and a.isascii()
    assert fontbuild._ps_name("Hand NC") == "Hand-NC"


def test_restart_marks_running_preview_as_interrupted(tmp_path, fonts_dir, cjk_font_path):
    class Slow(FakeGenerator):
        def generate(self, *args, **kwargs):
            time.sleep(2)
            return super().generate(*args, **kwargs)

    m1 = jobs.Manager(work_dir=tmp_path / "w", generator_factory=lambda s: Slow())
    m1.scanner.ready.wait(60)
    p = m1.create(layout(cjk_font_path, ["永和九年"]), "x.png")
    m1.submit_preview(p, {"refs": [p.data["default_ref"]], "content_font": f"{fonts_dir / cjk_font_path.name}#0",
                          "latin_font": "", "steps": 10, "seed": 1, "text": "永"})
    time.sleep(0.5)
    m2 = jobs.Manager(work_dir=tmp_path / "w", generator_factory=lambda s: Slow(), scanner=m1.scanner)
    assert m2.get(p.id).data["previews"][0]["status"] == "error"


# ---------- 字型分組與字型包 ----------

def _poster(rows):
    h = sum(int(size * 1.6) for _, _, size in rows) + 80
    img = Image.new("L", (1000, h), 255)
    d = ImageDraw.Draw(img)
    y = 40
    for text, font, size in rows:
        d.text((40, y), text, font=ImageFont.truetype(str(font), size), fill=0)
        y += int(size * 1.6)
    return img.convert("RGB")


def test_grouping_by_size_and_style(cjk_font_path):
    same = _poster([("永和九年歲在癸丑", cjk_font_path, 70), ("暮春之初會於會稽", cjk_font_path, 70),
                    ("山陰之蘭亭修禊事", cjk_font_path, 70)])
    seg = segment.segment(same)
    assert len(analyze.group_glyphs(seg)) == 1
    title = _poster([("蘭亭集序", cjk_font_path, 130), ("永和九年歲在癸丑暮春之初", cjk_font_path, 60),
                     ("會於會稽山陰之蘭亭修禊事", cjk_font_path, 60)])
    seg = segment.segment(title)
    groups = analyze.group_glyphs(seg)
    assert [len(g) for g in groups] == [4, 24]
    assert analyze.analyze(seg, groups[0])["glyph_count"] == 4


@pytest.mark.parametrize("base", [100, 300, 400, 500, 700, 900])
def test_weight_variants_stay_in_range(base):
    variants = analyze.weight_variants(base)
    assert any(w == round(base / 100) * 100 for _, w in variants)
    ref = analyze.stroke_for_weight(base)
    for _, w in variants:
        assert 0.5 <= analyze.stroke_for_weight(w) / ref <= 2.0


def test_package_has_all_weights_formats_and_specimens(tmp_path, fonts_dir, cjk_font_path):
    import zipfile

    from fontTools.ttLib import TTFont

    manager = jobs.Manager(work_dir=tmp_path / "work", generator_factory=lambda s: FakeGenerator())
    manager.scanner.ready.wait(60)
    c = TestClient(create_app(manager, require_setup=False))
    poster = _poster([("蘭亭集序", cjk_font_path, 130), ("永和九年歲在癸丑暮春之初", cjk_font_path, 60),
                      ("會於會稽山陰之蘭亭修禊事", cjk_font_path, 60)])
    p = c.post("/api/projects", files={"file": ("p.png", _encode(poster, "PNG"), "image/png")}).json()
    assert len(p["groups"]) == 2
    content = f"{fonts_dir / cjk_font_path.name}#0"
    fonts = [{"group": g["id"], "family": name, "refs": [g["default_ref"]]}
             for g, name in zip(p["groups"], ["Title", "內文"])]
    body = {"fonts": fonts, "preset": "custom", "custom": "我愛你", "latin": True, "content_font": content, "steps": 10}

    dup = c.post(f"/api/projects/{p['id']}/run", json={**body, "fonts": [fonts[0], {**fonts[1], "family": "Title"}]})
    assert dup.status_code == 409  # 名稱重複

    assert c.post(f"/api/projects/{p['id']}/run", json=body).status_code == 200
    end = time.time() + 300
    while time.time() < end:
        run = c.get(f"/api/projects/{p['id']}").json()["run"]
        if run["status"] in ("done", "error"):
            break
        time.sleep(0.3)
    assert run["status"] == "done", run
    zip_bytes = c.get(f"/api/projects/{p['id']}/package").content
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        for family in ("Title NC", "內文 NC"):
            entry = next(f for f in run["fonts"] if f["family"] == family)
            assert len(entry["outputs"]) >= 2
            for fmt in ("TTF", "OTF", "WOFF2"):
                assert sum(n.startswith(f"{family}/{fmt}/") for n in names) == len(entry["outputs"])
            for extra in ("fonts.css", "樣張.png", "全字表.pdf"):
                assert f"{family}/{extra}" in names
            # 各字重的 usWeightClass 與家族名稱
            weights = []
            for n in names:
                if n.startswith(f"{family}/OTF/"):
                    f = TTFont(io.BytesIO(zf.read(n)))
                    weights.append(f["OS/2"].usWeightClass)
                    typo = f["name"].getDebugName(16) or f["name"].getDebugName(1)
                    assert typo == family
                    assert "CFF " in f
            assert sorted(weights) == sorted(o["weight"] for o in entry["outputs"])
    # 試打用的單一檔案
    first = run["fonts"][0]["outputs"][0]["files"]["woff2"]
    assert c.get(f"/api/projects/{p['id']}/output/{first}").status_code == 200
    assert c.get(f"/api/projects/{p['id']}/output/../project.json").status_code == 404
