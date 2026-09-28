const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const KEY_BY_PRIMITIVE = { noul: "is_scam", score: "risk_level", choice: "scam_category" };

function toast(msg) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.remove("hidden");
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.add("hidden"), 4000);
}

function pct(x) { return (x * 100).toFixed(1) + "%"; }

function selectedPrimitives() {
  return $$(".primitive:checked").map((c) => c.value);
}
function selectedModel() {
  return $('input[name="model"]:checked').value;
}

function updateRunState() {
  const hasText = $("#text").value.trim().length > 0;
  const hasPrim = selectedPrimitives().length > 0;
  $("#run").disabled = !(hasText && hasPrim);
  $("#run-hint").textContent = !hasText
    ? "请输入文本"
    : !hasPrim
      ? "请至少勾选一种输出方式"
      : "";
}

async function loadHealth() {
  const badge = $("#health-badge");
  try {
    const r = await fetch("/api/health");
    const h = await r.json();
    if (h.ok) {
      badge.textContent = "模型就绪";
      badge.className = "badge badge-ok";
    } else {
      badge.textContent = "模型未加载";
      badge.className = "badge badge-err";
      toast("模型未加载: " + (h.load_error || "unknown"));
    }
  } catch (e) {
    badge.textContent = "服务不可用";
    badge.className = "badge badge-err";
  }
}

async function loadSamples() {
  const r = await fetch("/api/samples");
  const { samples } = await r.json();
  const box = $("#samples");
  box.innerHTML = "";
  for (const s of samples) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = s.label;
    b.onclick = () => {
      $("#text").value = s.text;
      updateRunState();
    };
    box.appendChild(b);
  }
}

let DEFAULT_QUESTIONS = null;

async function loadDefaults() {
  try {
    const r = await fetch("/api/defaults");
    if (r.ok) DEFAULT_QUESTIONS = (await r.json()).questions;
  } catch (e) {
    DEFAULT_QUESTIONS = null;
  }
}

function renderAdvanced() {
  const body = $("#advanced-body");
  body.innerHTML = "";
  if (!DEFAULT_QUESTIONS) {
    body.innerHTML = "<p class='hint'>默认问题不可用</p>";
    return;
  }
  for (const p of selectedPrimitives()) {
    const key = KEY_BY_PRIMITIVE[p];
    const q = DEFAULT_QUESTIONS[key];
    if (!q) continue;
    const div = document.createElement("div");
    div.className = "adv-item";
    const ins = (q.instructions || "").replace(/"/g, "&quot;");
    const crit = JSON.stringify(q.criteria === undefined ? [] : q.criteria);
    div.innerHTML =
      `<label>${p} · ${key} instructions</label>` +
      `<input data-key="${key}" data-field="instructions" value="${ins}" />` +
      `<label>criteria (JSON)</label>` +
      `<textarea data-key="${key}" data-field="criteria" rows="2">${crit}</textarea>`;
    body.appendChild(div);
  }
}

function collectAdvancedQuestions() {
  if (!DEFAULT_QUESTIONS) return null;
  const out = {};
  let edited = false;
  for (const p of selectedPrimitives()) {
    const key = KEY_BY_PRIMITIVE[p];
    if (!DEFAULT_QUESTIONS[key]) continue;
    const base = JSON.parse(JSON.stringify(DEFAULT_QUESTIONS[key]));
    const insEl = $(`input[data-key="${key}"][data-field="instructions"]`);
    const critEl = $(`textarea[data-key="${key}"][data-field="criteria"]`);
    if (insEl) {
      base.instructions = insEl.value;
      if (insEl.value !== DEFAULT_QUESTIONS[key].instructions) edited = true;
    }
    if (critEl) {
      try {
        const parsed = JSON.parse(critEl.value);
        base.criteria = parsed;
        if (JSON.stringify(parsed) !== JSON.stringify(DEFAULT_QUESTIONS[key].criteria)) edited = true;
      } catch (e) {
        toast(`${key}.criteria 不是合法 JSON，已用默认值`);
      }
    }
    out[key] = base;
  }
  return edited ? out : null;
}

function barRow(label, value, color) {
  const w = Math.max(0, Math.min(1, value)) * 100;
  return `<div class="bar-row">
    <span class="bar-label" title="${label}">${label}</span>
    <div class="bar-track"><div class="bar-fill" style="width:${w}%;background:${color}"></div></div>
    <span class="bar-pct">${pct(value)}</span>
  </div>`;
}

function renderNoul(ans) {
  const p = ans.noul;
  const isScam = p >= 0.5;
  const color = isScam ? "var(--red)" : "var(--green)";
  const verdict = isScam ? `🚨 诈骗 ${pct(p)}` : `✅ 正常 ${pct(1 - p)}`;
  return `<div class="result-card">
    <h3>是否诈骗 · noul</h3>
    <div class="verdict" style="color:${color}">${verdict}</div>
    ${barRow("false (正常)", ans.probabilities.false, "var(--green)")}
    ${barRow("true (诈骗)", ans.probabilities.true, "var(--red)")}
  </div>`;
}

function renderScore(ans) {
  const rows = Object.entries(ans.distribution)
    .map(([k, v]) => barRow(k, v, "var(--orange)"))
    .join("");
  return `<div class="result-card">
    <h3>风险评分 · score</h3>
    <div class="verdict">期望分 ${ans.score.toFixed(2)} / ${ans.max_score}</div>
    ${rows}
  </div>`;
}

function renderChoice(ans) {
  const sorted = Object.entries(ans.probabilities).sort((a, b) => b[1] - a[1]);
  const [topLabel, topProb] = sorted[0];
  const rows = sorted
    .map(([k, v], i) => barRow(k, v, i === 0 ? "var(--accent)" : "var(--border)"))
    .join("");
  return `<div class="result-card">
    <h3>诈骗类别 · choice</h3>
    <div class="verdict">🏷 ${topLabel}
      <span style="color:var(--muted);font-size:14px">${pct(topProb)}</span></div>
    ${rows}
  </div>`;
}

const RENDERERS = {
  is_scam: renderNoul,
  risk_level: renderScore,
  scam_category: renderChoice,
};

function renderResults(body) {
  const box = $("#results");
  box.innerHTML = "";
  for (const [key, ans] of Object.entries(body.answers)) {
    if (RENDERERS[key]) box.insertAdjacentHTML("beforeend", RENDERERS[key](ans));
  }
  $("#meta").textContent =
    `模型: ${body.routing.model} · 延迟: ${body.latency_ms.toFixed(0)}ms · 原因: ${body.routing.reason}`;
}

async function run() {
  const btn = $("#run");
  btn.disabled = true;
  btn.textContent = "分析中…";
  try {
    const payload = {
      text: $("#text").value,
      primitives: selectedPrimitives(),
      model: selectedModel(),
    };
    const adv = collectAdvancedQuestions();
    if (adv) payload.questions = adv;

    const r = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await r.json();
    if (!r.ok) {
      toast(body.detail || `HTTP ${r.status}`);
      return;
    }
    renderResults(body);
  } catch (e) {
    toast("请求失败: " + e.message);
  } finally {
    btn.textContent = "开始分析";
    updateRunState();
  }
}

async function init() {
  $("#text").addEventListener("input", updateRunState);
  $$(".primitive").forEach((c) =>
    c.addEventListener("change", () => {
      updateRunState();
      renderAdvanced();
    }),
  );
  $("#run").addEventListener("click", run);
  await loadHealth();
  await loadSamples();
  await loadDefaults();
  renderAdvanced();
  updateRunState();
}

init();