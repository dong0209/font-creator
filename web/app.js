"use strict";

const $ = (id) => document.getElementById(id);
// groups：各字型分組的設定 { [id]: { export, family, refs, latinFont } }；active：目前檢視的分組
const state = { status: null, project: null, groups: {}, active: null, polling: null };

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

const ncName = (name) => {
  const base = name.trim().replace(/\s+NC$/i, "");
  return base ? `${base} NC` : "";
};
const groupOf = (id) => (state.project.groups || []).find((g) => g.id === id) || state.project.groups[0];
const activeConf = () => state.groups[state.active];

// ---------- 初始狀態 ----------

async function loadStatus() {
  const s = await api("/api/status");
  state.status = s;
  const problems = [...s.problems];
  if (!s.scanning && !s.content_fonts.length) problems.push(`找不到中文字型：請把字型檔放到 ${s.fonts_dir}`);
  $("setup-list").replaceChildren(...problems.map((p) => el("li", {}, p)));
  $("setup").hidden = problems.length === 0;

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

  if (!document.querySelector('input[name="preset"]')) {
    $("presets").replaceChildren(...s.presets.map((p) => {
      const input = el("input", { type: "radio", name: "preset", value: p.id });
      if (p.id === "common") input.checked = true;
      input.addEventListener("change", onPresetChange);
      return el("label", {}, input, p.label);
    }));
  }
  onPresetChange();
  state.ready = !s.scanning && s.content_fonts.length > 0;
  $("preview-btn").disabled = !state.ready;
  if (state.project) renderRun(state.project);
}

async function loadHistory() {
  const items = await api("/api/projects");
  $("history-list").replaceChildren(...items.slice(0, 20).map((p) => {
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

function initGroups(p) {
  // 預設：3 個字以上的分組都匯出
  state.groups = {};
  for (const g of p.groups || []) {
    state.groups[g.id] = {
      export: g.glyphs.length >= 3 || p.groups.length === 1,
      family: "",
      refs: g.default_ref === null ? [] : [g.default_ref],
      latinFont: "",
    };
  }
  const run = p.run;
  if (run && run.fonts) {  // 還原上次的設定
    for (const conf of Object.values(state.groups)) conf.export = false;
    for (const f of run.fonts) {
      if (!state.groups[f.group]) continue;
      Object.assign(state.groups[f.group], {
        export: true, family: f.family.replace(/\s+NC$/i, ""), refs: [...f.refs], latinFont: f.latin_font || "",
      });
    }
  }
  const first = (p.groups || []).find((g) => state.groups[g.id].export) || (p.groups || [])[0];
  state.active = first ? first.id : null;
}

function showProject(p) {
  const fresh = !state.project || state.project.id !== p.id;
  state.project = p;
  history.replaceState(null, "", `#${p.id}`);
  if (fresh) {
    state.secPerGlyph = undefined;
    initGroups(p);
    if (p.run) restoreRunSettings(p.run);
    renderSource(p);
    renderGroups();
    selectGroup(state.active);
  }
  renderPreviews(p);
  renderRun(p);
  schedulePoll(p);
}

function restoreRunSettings(run) {
  const radio = document.querySelector(`input[name="preset"][value="${run.preset}"]`);
  if (radio) radio.checked = true;
  $("custom").value = run.custom || "";
  $("latin").checked = run.latin !== false;
  if (run.steps) $("steps").value = String(run.steps);
  if (run.seed !== undefined) $("seed").value = String(run.seed);
  if (run.content_font) $("content-font").value = run.content_font;
  onPresetChange();
}

function renderSource(p) {
  $("analysis").hidden = false;
  const img = $("source");
  img.src = `/api/projects/${p.id}/source`;
  img.onload = () => drawBoxes(p, img.naturalWidth, img.naturalHeight);
  const ok = (p.groups || []).length > 0 && p.analysis.ok;
  $("refs-card").hidden = !ok;
  $("run-card").hidden = !ok;
  if (!ok) {
    $("groups-wrap").hidden = true;
    $("style-name").textContent = p.analysis.message || "圖片中找不到文字";
    $("weight-name").textContent = "";
    $("style-note").textContent = "";
    $("features").replaceChildren();
    $("analysis-meta").textContent = "";
    $("analysis-warn").replaceChildren();
  }
}

function renderGroups() {
  const p = state.project;
  const groups = p.groups || [];
  $("groups-wrap").hidden = groups.length < 2;
  $("group-count").textContent = String(groups.length);
  $("groups").replaceChildren(...groups.map((g, n) => {
    const conf = state.groups[g.id];
    const check = el("input", { type: "checkbox", "aria-label": `匯出${g.name}` });
    check.checked = conf.export;
    check.addEventListener("click", (e) => e.stopPropagation());
    check.addEventListener("change", () => { conf.export = check.checked; renderFontList(); });
    const thumbs = [...g.glyphs].sort((a, b) => p.glyphs[b].score - p.glyphs[a].score).slice(0, 4)
      .map((i) => el("img", { src: `/api/projects/${p.id}/refs/${i}.png`, alt: "" }));
    const card = el("div", { class: `group-card c${n % 6}`, "data-id": g.id, role: "button", tabindex: "0" },
      el("div", { class: "group-head" }, check, el("strong", {}, g.name), el("span", { class: "small muted" }, "匯出")),
      el("div", { class: "group-thumbs" }, ...thumbs),
      el("p", { class: "small muted" }, `${g.glyphs.length} 字・${g.analysis.ok ? g.analysis.style : "無法分析"}`));
    card.addEventListener("click", () => selectGroup(g.id));
    card.addEventListener("keydown", (e) => { if (e.key === "Enter") selectGroup(g.id); });
    return card;
  }));
}

function selectGroup(id) {
  if (!id) return;
  state.active = id;
  const g = groupOf(id);
  const many = state.project.groups.length > 1;
  document.querySelectorAll(".group-card").forEach((c) => c.classList.toggle("active", c.dataset.id === id));
  renderAnalysis(g);
  renderRefs(g);
  $("refs-group").textContent = many ? g.name : "";
  $("refs-group").hidden = !many;
  loadLatin(state.project, g.id);
  renderFontList();
}

function renderAnalysis(g) {
  const a = g.analysis;
  if (!a.ok) {
    $("style-name").textContent = a.message;
    $("weight-name").textContent = "";
    $("features").replaceChildren();
    return;
  }
  $("style-name").textContent = a.style;
  $("weight-name").textContent = `字重 ${a.weight}（${a.weight_name}）`;
  $("style-note").textContent = a.style_note;
  $("features").replaceChildren(...a.features.map((f) =>
    el("tr", {}, el("td", {}, f.label), el("td", {}, f.value === null ? "—" : `${f.value} ${f.unit}`), el("td", {}, f.text))));
  const inv = state.project.analysis.inverted ? "（淺字深底，已自動反相）" : "";
  const scope = state.project.groups.length > 1 ? "此組" : "偵測到";
  $("analysis-meta").textContent = `${scope} ${a.glyph_count} 個字、${a.line_count} 行${inv}。`;
  $("analysis-warn").replaceChildren(...(a.warnings || []).map((w) => el("li", {}, w)));
}

function drawBoxes(p, w, h) {
  const img = $("source");
  w = w || img.naturalWidth;
  h = h || img.naturalHeight;
  if (!w || !h) return;
  // 切字是在縮圖後的座標上做的（長邊最多 3000px）
  const scale = Math.max(w, h) > 3000 ? 3000 / Math.max(w, h) : 1;
  const svg = $("boxes");
  svg.setAttribute("viewBox", `0 0 ${w * scale} ${h * scale}`);
  svg.setAttribute("preserveAspectRatio", "none");
  const order = (p.groups || []).map((g) => g.id);
  const refs = activeConf() ? activeConf().refs : [];
  svg.replaceChildren(...p.glyphs.map((g) => {
    const [x0, y0, x1, y1] = g.box;
    const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    r.setAttribute("x", x0); r.setAttribute("y", y0);
    r.setAttribute("width", x1 - x0); r.setAttribute("height", y1 - y0);
    const gi = Math.max(order.indexOf(g.group), 0);
    const dim = order.length > 1 && g.group !== state.active ? " dim" : "";
    r.setAttribute("class", `c${gi % 6}${dim}${refs.includes(g.index) ? " sel" : ""}`);
    return r;
  }));
}

function renderRefs(g) {
  const p = state.project;
  const sorted = [...g.glyphs].sort((a, b) => p.glyphs[b].score - p.glyphs[a].score);
  $("refs").replaceChildren(...sorted.map((i) => {
    const btn = el("button", { type: "button", title: `適合度 ${Math.round(p.glyphs[i].score * 100)}`, "data-index": i },
      el("img", { src: `/api/projects/${p.id}/refs/${i}.png`, alt: `參考字 ${i}` }));
    btn.addEventListener("click", () => toggleRef(i));
    return btn;
  }));
  updateRefs();
}

function toggleRef(index) {
  const max = (state.status && state.status.max_refs) || 3;
  const conf = activeConf();
  if (conf.refs.includes(index)) {
    if (conf.refs.length > 1) conf.refs = conf.refs.filter((i) => i !== index);
  } else if (conf.refs.length < max) {
    conf.refs.push(index);
  } else {
    conf.refs = [...conf.refs.slice(1), index];  // 已滿：換掉最早選的
  }
  updateRefs();
}

function updateRefs() {
  const conf = activeConf();
  if (!conf) return;
  document.querySelectorAll("#refs button").forEach((b) => {
    const order = conf.refs.indexOf(Number(b.dataset.index));
    b.classList.toggle("sel", order >= 0);
    b.dataset.order = order >= 0 ? String(order + 1) : "";
  });
  drawBoxes(state.project);
  const n = conf.refs.length;
  $("refs-count").textContent = n > 1
    ? `已選 ${n} 個參考字：每個字會生成 ${n} 次並自動挑最好的，時間約 ${n} 倍。`
    : "已選 1 個參考字。可再多選 1–2 個，品質更穩定，但時間會等比增加。";
  renderFontList();
}

// ---------- 英數字字型 ----------

async function loadLatin(p, groupId) {
  const sel = $("latin-font");
  try {
    const res = await api(`/api/projects/${p.id}/latin?group=${encodeURIComponent(groupId)}`);
    if (state.active !== groupId || state.project.id !== p.id) return;
    if (!res.ready) {
      sel.replaceChildren(el("option", { value: "" }, "比對字型中…"));
      setTimeout(() => { if (state.active === groupId) loadLatin(p, groupId); }, 3000);
      return;
    }
    const wanted = activeConf().latinFont;
    const options = res.candidates.map((c, i) => el("option", { value: c.id }, `${i === 0 ? "最接近：" : ""}${c.label}`));
    sel.replaceChildren(el("option", { value: "" }, "自動（最接近的字型）"), ...options);
    sel.value = [...sel.options].some((o) => o.value === wanted) ? wanted : "";
  } catch (_) {
    sel.replaceChildren(el("option", { value: "" }, "自動（最接近的字型）"));
  }
}
$("latin-font").addEventListener("change", () => { activeConf().latinFont = $("latin-font").value; });

// ---------- 試生成 ----------

function shared() {
  return {
    content_font: $("content-font").value,
    steps: Number($("steps").value),
    seed: Number($("seed").value) || 0,
  };
}

$("preview-btn").addEventListener("click", async () => {
  const p = state.project;
  if (!p) return;
  const conf = activeConf();
  try {
    await api(`/api/projects/${p.id}/preview`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...shared(), group: state.active, refs: conf.refs, latin_font: conf.latinFont,
        text: $("preview-text").value }),
    });
    openProject(p.id);
  } catch (err) {
    alert(err.message);
  }
});

function renderPreviews(p) {
  const groupName = (id) => (p.groups.length > 1 && id ? `${groupOf(id).name}・` : "");
  $("previews").replaceChildren(...(p.previews || []).map((v) => {
    const s = v.settings;
    const wrap = el("div", { class: "preview" });
    const head = el("p", { class: "small muted" },
      `${groupName(s.group)}參考字 ${(s.refs || []).map((r) => `#${r}`).join("、")}・${s.steps} 步・種子 ${s.seed}`);
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
      const useBtn = el("button", { type: "button" }, "用這組設定");
      useBtn.addEventListener("click", () => {
        if (s.group && state.groups[s.group]) {
          Object.assign(state.groups[s.group], { refs: [...(s.refs || [])], latinFont: s.latin_font || "", export: true });
          renderGroups();
          selectGroup(s.group);
        }
        $("steps").value = String(s.steps);
        $("seed").value = String(s.seed);
        $("content-font").value = s.content_font;
        state.secPerGlyph = per / Math.max((s.refs || [1]).length, 1);
        updateEstimate();
        $("run-card").scrollIntoView({ behavior: "smooth" });
      });
      wrap.append(useBtn);
      if (state.secPerGlyph === undefined && per) {
        state.secPerGlyph = per / Math.max((s.refs || [1]).length, 1);
        updateEstimate();
      }
    }
    return wrap;
  }));
}

const loadedFaces = new Set();
function ensureFontFace(family, url, descriptors = {}) {
  const key = `${family}|${url}`;
  if (loadedFaces.has(key)) return;
  loadedFaces.add(key);
  const face = new FontFace(family, `url(${url})`, descriptors);
  face.load().then((f) => document.fonts.add(f)).catch(() => loadedFaces.delete(key));
}

// ---------- 生成字型包 ----------

function exportedGroups() {
  return (state.project.groups || []).filter((g) => state.groups[g.id] && state.groups[g.id].export);
}

function renderFontList() {
  const p = state.project;
  if (!p) return;
  const list = exportedGroups();
  const many = p.groups.length > 1;
  $("font-list").replaceChildren(...(list.length ? list.map((g) => {
    const conf = state.groups[g.id];
    const input = el("input", { maxlength: "40", placeholder: "例如 MyHand" });
    input.value = conf.family;
    const final = el("strong", {}, ncName(conf.family) || "—");
    input.addEventListener("input", () => { conf.family = input.value; final.textContent = ncName(input.value) || "—"; });
    const meta = el("span", { class: "small muted" },
      `參考字 ${conf.refs.length} 個・${g.analysis.style || ""}・字重 ${g.analysis.weight || "—"}`);
    const pick = el("button", { type: "button", class: "link" }, "調整參考字");
    pick.addEventListener("click", () => { selectGroup(g.id); $("refs-card").scrollIntoView({ behavior: "smooth" }); });
    return el("div", { class: "font-row" },
      el("label", { class: "grow" }, many ? `${g.name} 的字型名稱` : "字型名稱", input),
      el("div", { class: "font-meta" }, el("span", { class: "muted" }, "實際名稱：", final), meta, many ? pick : ""));
  }) : [el("p", { class: "msg error" }, "請在上方勾選至少一組要匯出的字型。")]));
  updateEstimate();
}

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
  if (!state.project || !state.project.groups) return;
  const n = charCount();
  const list = exportedGroups();
  const passes = list.reduce((sum, g) => sum + Math.max(state.groups[g.id].refs.length, 1), 0);
  const what = `${list.length} 套字型 × 約 ${n.toLocaleString()} 字`;
  $("estimate").textContent = state.secPerGlyph
    ? `${what}，依試生成速度估計 AI 需 ${formatDuration(n * state.secPerGlyph * passes)}，另加描圖與打包（可隨時暫停、之後續跑）。`
    : `${what}。先試生成一次，就能估計所需時間。`;
}

$("run-btn").addEventListener("click", async () => {
  const p = state.project;
  const list = exportedGroups();
  if (!list.length) { alert("請至少勾選一組要匯出的字型"); return; }
  const fonts = [];
  for (const g of list) {
    const conf = state.groups[g.id];
    if (!conf.family.trim()) { alert(`請輸入「${g.name}」的字型名稱`); return; }
    if (!conf.refs.length) { alert(`「${g.name}」沒有參考字`); return; }
    fonts.push({ group: g.id, family: conf.family.trim(), refs: conf.refs, latin_font: conf.latinFont });
  }
  try {
    await api(`/api/projects/${p.id}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...shared(), fonts, preset: selectedPreset(), custom: $("custom").value, latin: $("latin").checked }),
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

const FONT_STATUS = { pending: "等待中", generating: "AI 生成中", generated: "已生成，等待描圖", building: "描圖與輸出中", done: "完成" };

function renderRun(p) {
  const run = p.run;
  const active = run && (run.status === "queued" || run.status === "running");
  $("run-btn").disabled = !!active || !state.ready;
  if (!run || !run.fonts) {  // 沒有生成紀錄（或舊版格式）
    $("progress").hidden = true;
    $("done").hidden = true;
    return;
  }
  $("progress").hidden = run.status === "done";
  $("done").hidden = run.status !== "done";

  const pct = run.total ? (100 * run.done) / run.total : 0;
  $("bar-fill").style.width = `${pct.toFixed(1)}%`;
  let text = `${runLabel(run)}：AI 生成 ${run.done.toLocaleString()} / ${run.total.toLocaleString()} 字（${pct.toFixed(1)}%）`;
  if (run.status === "running" && run.rate && run.done < run.total) {
    text += `，剩餘約 ${formatDuration((run.total - run.done) * run.rate)}`;
  }
  if (run.message) text += `\n${run.message}`;
  $("progress-text").textContent = text;
  $("progress-text").className = run.status === "error" ? "msg error" : "msg";
  $("font-status").replaceChildren(...run.fonts.map((f) => {
    const skipped = f.skipped && f.skipped.length ? `，內容字型缺 ${f.skipped.length} 字已略過` : "";
    return el("li", {}, `${f.family}：${FONT_STATUS[f.status] || f.status}（${f.done.toLocaleString()} / ${f.total.toLocaleString()}${skipped}）`);
  }));
  $("cancel-btn").hidden = !active;
  $("resume-btn").hidden = !(run.status === "paused" || run.status === "error");

  if (run.status === "done") renderDone(p, run);
}

function renderDone(p, run) {
  $("download").href = `/api/projects/${p.id}/package`;
  $("download").textContent = `下載字型包（${run.package}）`;
  const totalFiles = run.fonts.reduce((n, f) => n + f.outputs.length * 3, 0);
  $("done-note").textContent = `${run.fonts.length} 套字型、${totalFiles} 個字型檔。此字型禁止用於商業用途。`;
  const out = (rel) => `/api/projects/${p.id}/output/${rel.split("/").map(encodeURIComponent).join("/")}`;
  $("families").replaceChildren(...run.fonts.map((f) => {
    const css = `fam-${p.id}-${f.group}-${run.finished}`;
    for (const o of f.outputs) ensureFontFace(css, out(o.files.woff2), { weight: String(o.weight) });
    const weights = f.outputs.map((o) => `${o.style} ${o.weight}`).join("・");
    return el("div", { class: "family" },
      el("h3", {}, f.family),
      el("p", { class: "small muted" },
        `${f.glyph_count.toLocaleString()} 字・${weights}${f.latin_font_name ? `・英數字取自 ${f.latin_font_name}` : ""}`),
      el("a", { href: out(`${f.folder}/樣張.png`), target: "_blank" },
        el("img", { class: "specimen", src: out(`${f.folder}/樣張.png`), alt: `${f.family} 樣張` })),
      el("p", { class: "small" }, el("a", { href: out(`${f.folder}/全字表.pdf`), target: "_blank" }, "開啟全字表（PDF）")),
      ...f.outputs.map((o) => el("div", { class: "tryout-row" },
        el("span", { class: "small muted" }, `${o.style} ${o.weight}`),
        el("div", { class: "tryout", style: `font-family: "${css}", serif; font-weight: ${o.weight}` }))));
  }));
  renderTryout();
}

function renderTryout() {
  document.querySelectorAll("#families .tryout").forEach((n) => { n.textContent = $("tryout").value; });
}
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
