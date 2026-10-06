"use strict";

const $ = (id) => document.getElementById(id);
const state = { status: null, project: null, ref: null, polling: null, fontVersion: 0 };

async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch (_) { /* 非 JSON 錯誤 */ }
    throw new Error(detail);
  }
  return res.json();
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of children) node.append(c);
  return node;
}

function formatDuration(sec) {
  if (!isFinite(sec) || sec <= 0) return "—";
  const h = Math.floor(sec / 3600);
  const m = Math.round((sec % 3600) / 60);
  return h ? `${h} 小時 ${m} 分` : `${Math.max(m, 1)} 分鐘`;
}

function setMsg(id, text, error = false) {
  const node = $(id);
  node.textContent = text;
  node.classList.toggle("error", error);
}

// ---------- 初始狀態 ----------

async function loadStatus() {
  const s = await api("/api/status");
  state.status = s;
  const setup = $("setup");
  const list = $("setup-list");
  list.replaceChildren();
  const problems = [...s.problems];
  if (!s.scanning && !s.content_fonts.length) problems.push(`找不到中文字型：請把字型檔放到 ${s.fonts_dir}`);
  problems.forEach((p) => list.append(el("li", {}, p)));
  setup.hidden = problems.length === 0;

  const sel = $("content-font");
  const current = sel.value;
  sel.replaceChildren(...s.content_fonts.map((f) => el("option", { value: f.id },
    `${f.recommended ? "★ " : ""}${f.label}${f.coverage < 0.99 ? `（常用字 ${Math.round(f.coverage * 100)}%）` : ""}`)));
  if (current) sel.value = current;
  const hasRecommended = s.content_fonts.some((f) => f.recommended);
  $("content-hint").textContent = s.scanning ? "掃描字型中…"
    : hasRecommended ? "★ 為 FontDiffuser 訓練時使用的字型，效果最好；缺字會自動以其他字型補上。"
      : `建議下載「開心宋體 A」（KaiXinSongA.ttf，FontDiffuser 訓練時用的字型）放到 ${s.fonts_dir}，效果最好。`;
  if (s.scanning) setTimeout(loadStatus, 2000);
  if (!$("preview-text").value) $("preview-text").value = s.preview_text;

  const presets = $("presets");
  presets.replaceChildren(...s.presets.map((p, i) => {
    const input = el("input", { type: "radio", name: "preset", value: p.id });
    if (p.id === "common") input.checked = true;
    input.addEventListener("change", onPresetChange);
    return el("label", {}, input, p.label);
  }));
  onPresetChange();
  state.ready = !s.scanning && s.content_fonts.length > 0;
  $("preview-btn").disabled = !state.ready;
}

async function loadHistory() {
  const items = await api("/api/projects");
  const list = $("history-list");
  list.replaceChildren(...items.slice(0, 20).map((p) => {
    const when = new Date(p.created * 1000).toLocaleString("zh-TW");
    const run = p.run ? `（${runLabel(p.run)}）` : "";
    return el("li", {}, el("a", { href: `#${p.id}`, onclick: (e) => { e.preventDefault(); openProject(p.id); } },
      `${p.filename || "未命名"} · ${when} · ${p.glyphs} 字${run}`));
  }));
  $("history").hidden = items.length === 0;
}

function runLabel(run) {
  return { queued: "排隊中", running: "生成中", paused: "已暫停", error: "發生錯誤", done: "已完成" }[run.status] || run.status;
}

// ---------- 上傳 ----------

async function upload(file) {
  if (!file) return;
  setMsg("upload-msg", "上傳並分析中…");
  const form = new FormData();
  form.append("file", file);
  try {
    const project = await api("/api/projects", { method: "POST", body: form });
    setMsg("upload-msg", "");
    showProject(project);
    loadHistory();
  } catch (err) {
    setMsg("upload-msg", err.message, true);
  }
}

const drop = $("drop");
$("file").addEventListener("change", (e) => upload(e.target.files[0]));
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  upload(e.dataTransfer.files[0]);
});

// ---------- 專案顯示 ----------

async function openProject(id) {
  try {
    showProject(await api(`/api/projects/${id}`));
  } catch (err) {
    setMsg("upload-msg", err.message, true);
  }
}

function showProject(p) {
  const fresh = !state.project || state.project.id !== p.id;
  state.project = p;
  history.replaceState(null, "", `#${p.id}`);
  if (fresh) {
    state.refs = p.run && p.run.refs ? [...p.run.refs] : (p.default_ref === null ? [] : [p.default_ref]);
    state.latinWanted = p.run ? p.run.latin_font || "" : "";
    renderAnalysis(p);
    renderRefs(p);
    loadLatin(p);
    if (p.run) restoreRunSettings(p.run);
    onFamilyInput();
  }
  renderPreviews(p);
  renderRun(p);
  schedulePoll(p);
}

function restoreRunSettings(run) {
  if (run.family) $("family").value = run.family.replace(/\s+NC$/i, "");
  const radio = document.querySelector(`input[name="preset"][value="${run.preset}"]`);
  if (radio) radio.checked = true;
  $("custom").value = run.custom || "";
  $("latin").checked = run.latin !== false;
  if (run.steps) $("steps").value = String(run.steps);
  if (run.seed !== undefined) $("seed").value = String(run.seed);
  if (run.content_font) $("content-font").value = run.content_font;
  if (run.latin_font !== undefined) $("latin-font").value = run.latin_font;
  onPresetChange();
}

function renderAnalysis(p) {
  const a = p.analysis;
  $("analysis").hidden = false;
  const img = $("source");
  img.src = `/api/projects/${p.id}/source`;
  img.onload = () => drawBoxes(p, img.naturalWidth, img.naturalHeight);
  if (!a.ok) {
    $("style-name").textContent = a.message;
    $("weight-name").textContent = "";
    $("style-note").textContent = "";
    $("features").replaceChildren();
    $("refs-card").hidden = true;
    $("run-card").hidden = true;
    return;
  }
  $("style-name").textContent = a.style;
  $("weight-name").textContent = `字重 ${a.weight}（${a.weight_name}）`;
  $("style-note").textContent = a.style_note;
  $("features").replaceChildren(...a.features.map((f) =>
    el("tr", {}, el("td", {}, f.label), el("td", {}, f.value === null ? "—" : `${f.value} ${f.unit}`), el("td", {}, f.text))));
  $("analysis-meta").textContent = `偵測到 ${a.glyph_count} 個字、${a.line_count} 行${a.inverted ? "（淺字深底，已自動反相）" : ""}。`;
  $("refs-card").hidden = false;
  $("run-card").hidden = false;
}

function drawBoxes(p, w, h) {
  // 切字是在縮圖後的座標上做的（長邊最多 3000px）
  const scale = Math.max(w, h) > 3000 ? 3000 / Math.max(w, h) : 1;
  const svg = $("boxes");
  svg.setAttribute("viewBox", `0 0 ${w * scale} ${h * scale}`);
  svg.setAttribute("preserveAspectRatio", "none");
  svg.replaceChildren(...p.glyphs.map((g) => {
    const [x0, y0, x1, y1] = g.box;
    const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    r.setAttribute("x", x0); r.setAttribute("y", y0);
    r.setAttribute("width", x1 - x0); r.setAttribute("height", y1 - y0);
    r.dataset.index = g.index;
    if (state.refs.includes(g.index)) r.classList.add("sel");
    return r;
  }));
}

function renderRefs(p) {
  const sorted = [...p.glyphs].sort((a, b) => b.score - a.score);
  $("refs").replaceChildren(...sorted.map((g) => {
    const btn = el("button", { type: "button", title: `適合度 ${Math.round(g.score * 100)}`, "data-index": g.index },
      el("img", { src: `/api/projects/${p.id}/refs/${g.index}.png`, alt: `參考字 ${g.index}` }));
    btn.addEventListener("click", () => toggleRef(g.index));
    return btn;
  }));
  updateRefs();
}

function toggleRef(index) {
  const max = (state.status && state.status.max_refs) || 3;
  if (state.refs.includes(index)) {
    if (state.refs.length > 1) state.refs = state.refs.filter((i) => i !== index);
  } else if (state.refs.length < max) {
    state.refs.push(index);
  } else {
    state.refs = [...state.refs.slice(1), index];  // 已滿：換掉最早選的
  }
  updateRefs();
}

function setRefs(refs) {
  state.refs = [...refs];
  updateRefs();
}

function updateRefs() {
  updateEstimate();
  const on = (i) => state.refs.includes(Number(i));
  document.querySelectorAll("#refs button").forEach((b) => {
    b.classList.toggle("sel", on(b.dataset.index));
    const order = state.refs.indexOf(Number(b.dataset.index));
    b.dataset.order = order >= 0 ? String(order + 1) : "";
  });
  document.querySelectorAll("#boxes rect").forEach((r) => r.classList.toggle("sel", on(r.dataset.index)));
  const n = state.refs.length;
  $("refs-count").textContent = n > 1
    ? `已選 ${n} 個參考字：每個字會生成 ${n} 次並自動挑最好的，時間約 ${n} 倍。`
    : "已選 1 個參考字。可再多選 1–2 個，品質更穩定，但時間會等比增加。";
}

// ---------- 英數字字型 ----------

async function loadLatin(p) {
  const sel = $("latin-font");
  try {
    const res = await api(`/api/projects/${p.id}/latin`);
    if (!res.ready) {
      sel.replaceChildren(el("option", { value: "" }, "比對字型中…"));
      if (state.project && state.project.id === p.id) setTimeout(() => loadLatin(p), 3000);
      return;
    }
    const wanted = sel.value || state.latinWanted || "";
    const options = res.candidates.map((c, i) => el("option", { value: c.id }, `${i === 0 ? "最接近：" : ""}${c.label}`));
    sel.replaceChildren(el("option", { value: "" }, "自動（最接近的字型）"), ...options);
    sel.value = [...sel.options].some((o) => o.value === wanted) ? wanted : "";
  } catch (_) {
    sel.replaceChildren(el("option", { value: "" }, "自動（最接近的字型）"));
  }
}

// ---------- 試生成 ----------

function genSettings() {
  return {
    refs: state.refs,
    content_font: $("content-font").value,
    latin_font: $("latin-font").value,
    steps: Number($("steps").value),
    seed: Number($("seed").value) || 0,
  };
}

$("preview-btn").addEventListener("click", async () => {
  const p = state.project;
  if (!p) return;
  try {
    await api(`/api/projects/${p.id}/preview`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...genSettings(), text: $("preview-text").value }),
    });
    openProject(p.id);
  } catch (err) {
    alert(err.message);
  }
});

function renderPreviews(p) {
  const box = $("previews");
  const refName = (s) => `參考字 ${(s.refs || []).map((r) => `#${r}`).join("、")}・${s.steps} 步・種子 ${s.seed}`;
  box.replaceChildren(...(p.previews || []).map((v) => {
    const wrap = el("div", { class: "preview" });
    const head = el("p", { class: "small muted" }, refName(v.settings));
    wrap.append(head);
    if (v.status === "queued" || v.status === "running") {
      wrap.append(el("p", {}, v.status === "queued" ? "排隊中…" : (v.message || "生成中…")));
    } else if (v.status === "error") {
      wrap.append(el("p", { class: "msg error" }, v.message));
    } else {
      const base = `/api/projects/${p.id}/previews/${v.id}`;
      const family = `preview-${v.id}`;
      ensureFontFace(family, `${base}/preview.ttf`);
      wrap.append(
        el("div", { class: "preview-imgs" }, ...v.chars.map((c) =>
          el("img", { src: `${base}/${c.codePointAt(0).toString(16).toUpperCase()}.png`, alt: c, title: c }))),
        Object.assign(el("div", { class: "preview-text" }, v.chars.join("") + (v.latin || []).join("")),
          { style: `font-family: "${family}", serif` }),
      );
      if (v.latin_font) wrap.append(el("p", { class: "small muted" }, `英數字字型：${v.latin_font}`));
      const per = v.per_glyph || 0;
      if (per) head.append(`・每字約 ${per.toFixed(2)} 秒`);
      const useBtn = el("button", { type: "button" }, "用這組設定生成完整字型");
      useBtn.addEventListener("click", () => {
        setRefs(v.settings.refs || []);
        $("latin-font").value = v.settings.latin_font || "";
        $("steps").value = String(v.settings.steps);
        $("seed").value = String(v.settings.seed);
        $("content-font").value = v.settings.content_font;
        state.secPerGlyph = per / Math.max((v.settings.refs || [1]).length, 1);
        updateEstimate();
        $("run-card").scrollIntoView({ behavior: "smooth" });
      });
      wrap.append(useBtn);
      if (state.secPerGlyph === undefined && per) {
        state.secPerGlyph = per / Math.max((v.settings.refs || [1]).length, 1);
        updateEstimate();
      }
    }
    return wrap;
  }));
}

const loadedFaces = new Set();
function ensureFontFace(family, url) {
  if (loadedFaces.has(family)) return;
  loadedFaces.add(family);
  const face = new FontFace(family, `url(${url})`);
  face.load().then((f) => document.fonts.add(f)).catch(() => loadedFaces.delete(family));
}

// ---------- 正式生成 ----------

function onFamilyInput() {
  const name = $("family").value.trim().replace(/\s+NC$/i, "");
  $("family-final").textContent = name ? `${name} NC` : "—";
}
$("family").addEventListener("input", onFamilyInput);

function selectedPreset() {
  const checked = document.querySelector('input[name="preset"]:checked');
  return checked ? checked.value : "common";
}

function onPresetChange() {
  $("custom-wrap").hidden = selectedPreset() !== "custom";
  updateEstimate();
}
$("custom").addEventListener("input", updateEstimate);
$("latin").addEventListener("change", updateEstimate);

function charCount() {
  const preset = selectedPreset();
  const base = { common: 5401, "common+less": 13053, custom: 0 }[preset] || 0;
  const custom = preset === "custom" ? new Set([...$("custom").value].filter((c) => c.trim())).size : 0;
  return base + custom + 30;  // 英數字取自現成字型，不需 AI 生成
}

function updateEstimate() {
  const n = charCount();
  const refs = Math.max((state.refs || []).length, 1);
  const text = state.secPerGlyph
    ? `AI 需生成約 ${n.toLocaleString()} 字（${refs} 個參考字），依試生成速度估計需 ${formatDuration(n * state.secPerGlyph * refs)}（可隨時暫停、之後續跑）。`
    : `AI 需生成約 ${n.toLocaleString()} 字。先試生成一次，就能估計所需時間。`;
  $("estimate").textContent = text;
}

$("run-btn").addEventListener("click", async () => {
  const p = state.project;
  const family = $("family").value.trim();
  if (!family) { alert("請輸入字型名稱"); return; }
  try {
    await api(`/api/projects/${p.id}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...genSettings(), family, preset: selectedPreset(), custom: $("custom").value, latin: $("latin").checked,
      }),
    });
    openProject(p.id);
  } catch (err) {
    alert(err.message);
  }
});

$("cancel-btn").addEventListener("click", () => runAction("cancel"));
$("resume-btn").addEventListener("click", () => runAction("resume"));

async function runAction(action) {
  try {
    await api(`/api/projects/${state.project.id}/${action}`, { method: "POST" });
    openProject(state.project.id);
  } catch (err) {
    alert(err.message);
  }
}

function renderRun(p) {
  const run = p.run;
  const active = run && (run.status === "queued" || run.status === "running");
  $("run-btn").disabled = !!active || !state.ready;
  $("progress").hidden = !run || run.status === "done";
  $("done").hidden = !run || run.status !== "done";
  if (!run) return;

  const pct = run.total ? (100 * run.done) / run.total : 0;
  $("bar-fill").style.width = `${pct.toFixed(1)}%`;
  let text = `${runLabel(run)}：${run.done.toLocaleString()} / ${run.total.toLocaleString()} 字（${pct.toFixed(1)}%）`;
  if (run.status === "running" && run.rate) text += `，剩餘約 ${formatDuration((run.total - run.done) * run.rate)}`;
  if (run.message) text += `\n${run.message}`;
  if (run.skipped && run.skipped.length) text += `\n內容字型缺少 ${run.skipped.length} 字，已略過。`;
  $("progress-text").textContent = text;
  $("progress-text").className = run.status === "error" ? "msg error" : "msg";
  $("cancel-btn").hidden = !active;
  $("resume-btn").hidden = !(run.status === "paused" || run.status === "error");

  if (run.status === "done") {
    $("download").href = `/api/projects/${p.id}/font`;
    $("done-note").textContent = `「${run.family}」共 ${run.glyph_count.toLocaleString()} 字。此字型禁止用於商業用途。`;
    const family = `final-${p.id}-${run.finished}`;
    ensureFontFace(family, `/api/projects/${p.id}/font?inline=true&v=${run.finished}`);
    $("tryout-view").style.fontFamily = `"${family}", serif`;
    renderTryout();
  }
}

function renderTryout() { $("tryout-view").textContent = $("tryout").value; }
$("tryout").addEventListener("input", renderTryout);

// ---------- 輪詢 ----------

function schedulePoll(p) {
  clearTimeout(state.polling);
  const busy = (p.run && (p.run.status === "queued" || p.run.status === "running"))
    || (p.previews || []).some((v) => v.status === "queued" || v.status === "running");
  if (busy) state.polling = setTimeout(() => openProject(p.id), 2000);
}

// ---------- 啟動 ----------

window.addEventListener("hashchange", () => {
  const id = location.hash.slice(1);
  if (id && (!state.project || state.project.id !== id)) openProject(id);
});

(async () => {
  try {
    await loadStatus();
    await loadHistory();
    const id = location.hash.slice(1);
    if (id) openProject(id);
  } catch (err) {
    setMsg("upload-msg", `無法連線到伺服器：${err.message}`, true);
  }
})();
