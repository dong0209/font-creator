import io
import time

import pytest
from fastapi.testclient import TestClient
from fontTools.pens.boundsPen import BoundsPen
from fontTools.ttLib import TTFont

from fontmaker import analyze, charsets, fontbuild, jobs, segment
from fontmaker.content import ContentFont
from fontmaker.server import create_app

from .conftest import FakeGenerator, draw_text


def test_charsets():
    assert len(charsets.common()) == 5401
    assert len(charsets.less_common()) == 7652
    assert "的" in charsets.common()
    chars = charsets.build_charset("custom", "我我你 A", include_latin=False)
    assert chars[-3:] == ["我", "你", "A"] and " " not in chars
    assert len(chars) == len(set(chars))
    assert len(charsets.build_charset("common+less")) > 13053


@pytest.mark.parametrize("name, expected", [
    ("MyHand", "MyHand NC"),
    ("MyHand NC", "MyHand NC"),
    ("my hand nc", "my hand NC"),
    ("  手寫  體 ", "手寫 體 NC"),
    ("", "Generated NC"),
])
def test_nc_name(name, expected):
    assert fontbuild.nc_family_name(name) == expected


def test_segment_counts_cjk_chars(cjk_font_path):
    img = draw_text(cjk_font_path, ["永東國酬愛鬱", "川明好二三小"])
    seg = segment.segment(img)
    assert len(seg.lines) == 2
    assert [len(ids) for ids in seg.lines] == [6, 6]
    assert not seg.inverted
    ref = segment.style_image(seg.glyphs[0])
    assert ref.size == (96, 96)


def test_segment_inverted(cjk_font_path):
    seg = segment.segment(draw_text(cjk_font_path, ["永東國酬"], invert=True))
    assert seg.inverted
    assert len(seg.glyphs) == 4


def test_analysis_detects_slant_and_weight(cjk_font_path):
    upright = analyze.analyze(segment.segment(draw_text(cjk_font_path, ["永東國酬愛鬱", "靈鷹袋明好書"])))
    assert upright["ok"]
    assert abs(upright["slant"]) < 3
    assert 100 <= upright["weight"] <= 900
    keys = {f["key"] for f in upright["features"]}
    assert {"stroke", "contrast", "slant", "aspect", "roundness", "irregularity"} <= keys

    slanted = analyze.analyze(segment.segment(draw_text(cjk_font_path, ["永東國酬愛鬱"], shear=0.25)))
    assert slanted["slant"] > 8


def test_analysis_empty_image():
    from PIL import Image
    report = analyze.analyze(segment.segment(Image.new("RGB", (200, 100), "white")))
    assert not report["ok"]


def _glyph_bounds(font: TTFont, ch: str):
    name = font.getBestCmap()[ord(ch)]
    pen = BoundsPen(font.getGlyphSet())
    font.getGlyphSet()[name].draw(pen)
    return pen.bounds


def test_font_roundtrip_preserves_position(tmp_path, cjk_font_path):
    cf = ContentFont(cjk_font_path)
    sources = []
    for ch in "永國gA":
        img, placement = cf.render(ch)
        img = img.resize((96, 96))
        sources.append(fontbuild.GlyphSource(ch, img, placement))
    info = fontbuild.build_font(fontbuild.FontSpec(family="Test", glyphs=sources), tmp_path / "t.ttf")
    assert info["family"] == "Test NC"
    font = TTFont(tmp_path / "t.ttf")
    assert font["name"].getDebugName(1) == "Test NC"

    src = TTFont(str(cjk_font_path), fontNumber=0)
    upm = src["head"].unitsPerEm
    # 內容字型依「國」正規化：輸出單位 = 原始單位 / upm × 字級 / 128 × 1000
    k = 1000 / upm * cf.size / 128
    for ch in "永gA":
        got = _glyph_bounds(font, ch)
        ref = [v * k for v in _glyph_bounds(src, ch)]
        if ch == "永":  # 全形字會把字身框中心對齊到輸出字型的字身框中心
            dy = (fontbuild.ASCENT + fontbuild.DESCENT) / 2 - cf.em_cy * 1000
            ref = [ref[0], ref[1] + dy, ref[2], ref[3] + dy]
        # 96px 描圖的誤差約 1–2 個像素（1px ≈ 10 units）
        assert abs(got[1] - ref[1]) < 30, (ch, got, ref)
        assert abs(got[3] - ref[3]) < 30, (ch, got, ref)
    assert _glyph_bounds(font, "g")[1] < -50  # g 的下伸部仍在基線下
    guo_bounds = _glyph_bounds(font, "國")
    assert abs((guo_bounds[0] + guo_bounds[2]) / 2 - 500) < 30  # 全形字水平置中
    assert abs((guo_bounds[1] + guo_bounds[3]) / 2 - 380) < 40  # 垂直置中於字身框
    assert 850 < max(guo_bounds[2] - guo_bounds[0], guo_bounds[3] - guo_bounds[1]) < 960
    hmtx = font["hmtx"]
    assert hmtx[font.getBestCmap()[ord("永")]][0] == 1000


def _wait(client, pid, pred, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        data = client.get(f"/api/projects/{pid}").json()
        if pred(data):
            return data
        time.sleep(0.2)
    raise AssertionError(f"timeout: {data}")


def _content_id(manager, fonts_dir, cjk_font_path) -> str:
    manager.scanner.ready.wait(60)
    return f"{fonts_dir / cjk_font_path.name}#0"


def test_api_end_to_end(tmp_path, fonts_dir, cjk_font_path):
    fake = FakeGenerator()
    manager = jobs.Manager(work_dir=tmp_path / "work", generator_factory=lambda steps: fake)
    client = TestClient(create_app(manager, require_setup=False))
    content_id = _content_id(manager, fonts_dir, cjk_font_path)

    status = client.get("/api/status").json()
    fonts = status["content_fonts"]
    assert fonts[0]["id"] == content_id and fonts[0]["user"]  # 使用者放的字型排最前面

    buf = io.BytesIO()
    draw_text(cjk_font_path, ["永東國酬愛鬱"]).save(buf, "PNG")
    res = client.post("/api/projects", files={"file": ("sample.png", buf.getvalue(), "image/png")})
    assert res.status_code == 200, res.text
    project = res.json()
    pid, ref = project["id"], project["default_ref"]
    assert project["analysis"]["ok"] and len(project["glyphs"]) == 6
    assert client.get(f"/api/projects/{pid}/refs/{ref}.png").status_code == 200
    refs = [g["index"] for g in project["glyphs"]][:2]

    manager.scanner.latin_ready.wait(120)
    latin_res = client.get(f"/api/projects/{pid}/latin").json()
    assert latin_res["ready"] and latin_res["candidates"]

    bad = client.post(f"/api/projects/{pid}/preview", json={"refs": [ref], "content_font": "../x.ttf"})
    assert bad.status_code == 400
    bad = client.post(f"/api/projects/{pid}/preview", json={"refs": [0, 1, 2, 3], "content_font": content_id})
    assert bad.status_code == 422

    res = client.post(f"/api/projects/{pid}/preview", json={"refs": refs, "content_font": content_id, "text": "永A"})
    assert res.status_code == 200
    data = _wait(client, pid, lambda d: d["previews"][0]["status"] in ("done", "error"))
    preview = data["previews"][0]
    assert preview["status"] == "done", preview
    assert preview["chars"] == ["永"] and preview["latin"] == ["A"] and preview["latin_font"]
    assert client.get(f"/api/projects/{pid}/previews/{preview['id']}/preview.ttf").status_code == 200
    assert client.get(f"/api/projects/{pid}/previews/{preview['id']}/{ord('永'):X}.png").status_code == 200

    body = {"family": "測試 Hand", "preset": "custom", "custom": "我愛你", "latin": True,
            "refs": refs, "content_font": content_id, "steps": 10}
    res = client.post(f"/api/projects/{pid}/run", json=body)
    assert res.status_code == 200, res.text
    data = _wait(client, pid, lambda d: d["run"]["status"] in ("done", "error"))
    run = data["run"]
    assert run["status"] == "done", run
    assert run["family"] == "測試 Hand NC"
    assert run["total"] < 100  # 英數字不經 AI 生成

    res = client.get(f"/api/projects/{pid}/font")
    assert res.status_code == 200
    font = TTFont(io.BytesIO(res.content))
    cmap = font.getBestCmap()
    for ch in "我愛你，。AZaz09!~":
        assert ord(ch) in cmap, ch
    assert font["name"].getDebugName(1) == "測試 Hand NC"
    a_bounds = _glyph_bounds(font, "A")
    assert a_bounds[1] > -20 and 500 < a_bounds[3] < 900  # A 站在基線上、高度合理

    # 再跑一次同設定：字圖已快取，不會再呼叫生成器
    calls = fake.calls
    client.post(f"/api/projects/{pid}/run", json=body)
    _wait(client, pid, lambda d: d["run"]["status"] == "done" and d["run"].get("finished", 0) > run["finished"] - 1
          and d["run"]["done"] == d["run"]["total"])
    assert fake.calls == calls


def test_pause_and_resume(tmp_path, fonts_dir, cjk_font_path):
    class SlowFake(FakeGenerator):
        batch_size = 2

        def generate(self, contents, style, seed=123):
            time.sleep(0.2)
            return super().generate(contents, style, seed)

    manager = jobs.Manager(work_dir=tmp_path / "work", generator_factory=lambda steps: SlowFake())
    p = manager.create(draw_text(cjk_font_path, ["永東國"]), "x.png")
    settings = {"family": "P", "preset": "custom", "custom": "一二三四五六七八九十", "latin": False,
                "refs": [p.data["default_ref"]], "content_font": _content_id(manager, fonts_dir, cjk_font_path),
                "steps": 10, "seed": 1}
    manager.submit_run(p, settings)
    end = time.time() + 30
    while p.data["run"]["status"] != "running" and time.time() < end:
        time.sleep(0.05)
    manager.cancel(p)
    while p.data["run"]["status"] == "running" and time.time() < end:
        time.sleep(0.05)
    assert p.data["run"]["status"] == "paused"
    assert p.data["run"]["done"] < p.data["run"]["total"]

    # 模擬伺服器重啟：重新載入既有專案後續跑
    manager2 = jobs.Manager(work_dir=tmp_path / "work", generator_factory=lambda steps: SlowFake(),
                            scanner=manager.scanner)
    p2 = manager2.get(p.id)
    manager2.resume(p2)
    while p2.data["run"]["status"] != "done" and time.time() < end + 30:
        time.sleep(0.1)
    assert p2.data["run"]["status"] == "done"
    assert p2.data["run"]["glyph_count"] >= 10


def test_roundness_separates_square_and_rounded(cjk_font_path):
    import cv2
    import numpy as np

    seg = segment.segment(draw_text(cjk_font_path, ["永東國酬愛鬱"], size=120))
    square, rounded = [], []
    for g in seg.glyphs:
        sw = analyze._stroke_width(g.ink)
        r = max(3, int(sw * 0.9)) | 1
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
        ink = np.pad(g.ink.astype(np.uint8), r)
        soft = cv2.morphologyEx(cv2.morphologyEx(ink, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
        square.append(analyze._roundness(g.ink, sw))
        rounded.append(analyze._roundness(soft, sw))
    assert np.median(square) < 0.3
    assert np.median(rounded) > 0.6
