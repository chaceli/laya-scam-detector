// In-browser inference for the fine-tuned Laya scam detector.
//
// Everything runs client-side: onnxruntime-web (WASM) executes the fp16 ONNX
// graph and transformers.js tokenizes with the mmBERT 256k-vocab tokenizer.
// Text never leaves the browser.

// fp16, not int8: onnxruntime-web's WASM int8 kernels diverge from native
// ORT (P(scam) 0.194 vs 0.016 on the same input and weights), while fp16
// matches native to ~1e-5. Overridable from index.html (window.__MODEL_BASE)
// to self-host the weights.
const MODEL_BASE = window.__MODEL_BASE
  || "https://huggingface.co/LiChace/laya-scam-detector-onnx/resolve/main";
const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/ort.wasm.min.mjs";
const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6";

// mmBERT special-token ids (see src/laya_onnx.py::_resolve_special_tokens)
const CLS = 2, SEP = 1, MASK = 4;
const MAX_LEN = 512, HEAD_MAX_LEN = 192;

const QTYPE = { noul: 2, score: 1, choice: 0 };
const KEY_BY_PRIMITIVE = { noul: "is_scam", score: "risk_level", choice: "scam_category" };

// Mirrors schemas/scam.json. Editable from the Advanced panel.
const SCHEMA = {
  is_scam: {
    type: "noul",
    instructions: "Is this message a scam, fraud, phishing, or social engineering attempt?",
  },
  risk_level: {
    type: "score",
    instructions: "How high is the risk that this is malicious?",
    criteria: [
      "1 - clearly benign (normal message)",
      "2 - mildly suspicious (some red flags)",
      "3 - likely scam (multiple fraud signals)",
      "4 - high confidence scam (typical fraud pattern)",
      "5 - definitive fraud (obvious scam)",
    ],
  },
  scam_category: {
    type: "choice",
    instructions: "What category of scam does this most resemble?",
    criteria: {
      benign: "normal legitimate message (no fraud signal)",
      phishing: "phishing link, credential theft, account verification",
      crypto_scam: "cryptocurrency fraud, fake coin offering",
      investment_scam: "fake returns, stock tips, Ponzi scheme",
      lottery_scam: "fake prize, lucky draw, congratulations winner",
      job_scam: "fake job offer, upfront fee, mule recruitment",
      loan_scam: "fake loan offer, predatory lending",
      impersonation: "fake police, government, bank, customer service",
      romance_scam: "pig butchering, emotional manipulation, long game",
      delivery_fraud: "fake courier, refund, lost package",
      marketing: "legitimate but aggressive sales / promotions",
      adult_content: "adult, escort, sexual services",
      spam_general: "other unsolicited junk / noise",
    },
  },
};

const SAMPLES = [
  { label: "中文·快递理赔诈骗",
    text: "您好，我是XX快递客服，您有一个包裹在运输途中丢失，请点击链接填写个人信息进行理赔。" },
  { label: "中文·家人问候", text: "妈，我今晚回家吃饭，大概6点到家。" },
  { label: "英文·PayPal 钓鱼",
    text: "URGENT: Your PayPal account has been limited. Click here to verify: http://paypa1-secure.tk/login" },
  { label: "英文·会议确认", text: "Hi, just confirming our meeting at 3pm tomorrow. Looking forward to it." },
];

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const pct = (x) => (x * 100).toFixed(1) + "%";

function toast(msg) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.remove("hidden");
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.add("hidden"), 5000);
}

function selectedPrimitives() {
  return $$(".primitive:checked").map((c) => c.value);
}

function updateRunState() {
  const hasText = $("#text").value.trim().length > 0;
  const hasPrim = selectedPrimitives().length > 0;
  const ready = window.__MODEL_READY === true;
  $("#run").disabled = !(hasText && hasPrim && ready);
  let hint = "";
  if (!ready) hint = "模型加载中…";
  else if (!hasText) hint = "请输入文本";
  else if (!hasPrim) hint = "请至少勾选一种输出方式";
  $("#run-hint").textContent = hint;
}

// Mirrors src/laya_onnx.py::_tokenize_question
function tokenizeQuestion(rawEncode, question, options, qtype) {
  // The literal prefix is "choice question: " for every primitive, matching
  // src/laya_onnx.py::_tokenize_question. Using the primitive name instead
  // flips benign inputs to scam.
  const ids = [CLS, ...rawEncode(`choice question: ${question}`), SEP];
  const positions = [];
  const budget = Math.max(8, Math.floor((HEAD_MAX_LEN - 16) / options.length));
  for (const [label, desc] of options) {
    positions.push(ids.length);
    ids.push(MASK);
    ids.push(...rawEncode(` ${label}: ${desc}`).slice(0, budget));
  }
  ids.push(SEP);
  return [ids, positions];
}

function optionsFor(q, qtype) {
  if (qtype === QTYPE.noul) return [["false", "no"], ["true", "yes"]];
  if (qtype === QTYPE.score) return q.criteria.map((c, i) => [`level_${i}`, String(c)]);
  return Object.entries(q.criteria).map(([k, v]) => [String(k), String(v)]);
}

function softmax(logits) {
  const mx = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - mx));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

async function predictOne(session, ort, rawEncode, state, q, qtype) {
  const options = optionsFor(q, qtype);
  const [headIds, positions] = tokenizeQuestion(rawEncode, q.instructions, options, qtype);
  const room = Math.max(0, MAX_LEN - headIds.length - 1);
  const ids = [...headIds, ...rawEncode(state).slice(0, room), SEP];

  const n = positions.length;
  const i64 = (arr) =>
    new ort.Tensor("int64", BigInt64Array.from(arr.map(BigInt)), [1, arr.length]);
  const feeds = {
    input_ids: i64(ids),
    attention_mask: i64(ids.map(() => 1)),
    marker_pos: i64(positions),
    marker_mask: new ort.Tensor("bool", Uint8Array.from(positions.map(() => 1)), [1, n]),
    // qtype is rank 1 ([batch]), unlike the per-token tensors above.
    qtype: new ort.Tensor("int64", BigInt64Array.from([BigInt(qtype)]), [1]),
  };

  const t0 = performance.now();
  const out = await session.run(feeds);
  const ms = performance.now() - t0;
  const probs = softmax(Array.from(out.logits.data).slice(0, n).map(Number));

  if (qtype === QTYPE.noul) {
    const pTrue = probs[1] ?? probs[0];
    return { ms, answer: { noul: pTrue,
      probabilities: { false: probs[0], true: pTrue },
      confidence: Math.max(probs[0], pTrue) } };
  }
  if (qtype === QTYPE.score) {
    const dist = {};
    options.forEach((_, i) => { dist[q.criteria[i]] = probs[i]; });
    return { ms, answer: {
      score: probs.reduce((a, p, i) => a + p * i, 0),
      max_score: n - 1, distribution: dist, confidence: Math.max(...probs) } };
  }
  const labels = options.map(([l]) => l);
  const dist = {};
  labels.forEach((l, i) => { dist[l] = probs[i]; });
  const best = probs.indexOf(Math.max(...probs));
  return { ms, answer: { choice: labels[best], probabilities: dist, confidence: probs[best] } };
}

function barRow(label, value, color) {
  const w = Math.max(0, Math.min(1, value)) * 100;
  return `<div class="bar-row">
    <span class="bar-label" title="${label}">${label}</span>
    <div class="bar-track"><div class="bar-fill" style="width:${w}%;background:${color}"></div></div>
    <span class="bar-pct">${pct(value)}</span>
  </div>`;
}

function renderNoul(a) {
  const scam = a.noul >= 0.5;
  return `<div class="result-card">
    <h3>是否诈骗 · noul</h3>
    <div class="verdict" style="color:${scam ? "var(--red)" : "var(--green)"}">
      ${scam ? "🚨 诈骗 " + pct(a.noul) : "✅ 正常 " + pct(1 - a.noul)}</div>
    ${barRow("false (正常)", a.probabilities.false, "var(--green)")}
    ${barRow("true (诈骗)", a.probabilities.true, "var(--red)")}
  </div>`;
}

function renderScore(a) {
  const rows = Object.entries(a.distribution)
    .map(([k, v]) => barRow(k, v, "var(--orange)")).join("");
  return `<div class="result-card">
    <h3>风险评分 · score</h3>
    <div class="verdict">期望分 ${a.score.toFixed(2)} / ${a.max_score}</div>${rows}</div>`;
}

function renderChoice(a) {
  const sorted = Object.entries(a.probabilities).sort((x, y) => y[1] - x[1]);
  const [top, topP] = sorted[0];
  const rows = sorted.map(([k, v], i) =>
    barRow(k, v, i === 0 ? "var(--accent)" : "var(--border)")).join("");
  return `<div class="result-card">
    <h3>诈骗类别 · choice</h3>
    <div class="verdict">🏷 ${top}
      <span style="color:var(--muted);font-size:14px">${pct(topP)}</span></div>${rows}</div>`;
}

const RENDERERS = { is_scam: renderNoul, risk_level: renderScore, scam_category: renderChoice };

function renderAdvanced() {
  const body = $("#advanced-body");
  body.innerHTML = "";
  for (const p of selectedPrimitives()) {
    const key = KEY_BY_PRIMITIVE[p];
    const q = SCHEMA[key];
    if (!q) continue;
    const div = document.createElement("div");
    div.className = "adv-item";
    const crit = JSON.stringify(q.criteria === undefined ? [] : q.criteria);
    div.innerHTML =
      `<label>${p} · ${key} instructions</label>` +
      `<input data-key="${key}" data-field="instructions" value="${(q.instructions || "").replace(/"/g, "&quot;")}" />` +
      `<label>criteria (JSON)</label>` +
      `<textarea data-key="${key}" data-field="criteria" rows="2">${crit}</textarea>`;
    body.appendChild(div);
  }
}

function editedQuestions() {
  const out = {};
  let edited = false;
  for (const p of selectedPrimitives()) {
    const key = KEY_BY_PRIMITIVE[p];
    if (!SCHEMA[key]) continue;
    const base = JSON.parse(JSON.stringify(SCHEMA[key]));
    const insEl = $(`input[data-key="${key}"][data-field="instructions"]`);
    const critEl = $(`textarea[data-key="${key}"][data-field="criteria"]`);
    if (insEl) {
      base.instructions = insEl.value;
      if (insEl.value !== SCHEMA[key].instructions) edited = true;
    }
    if (critEl) {
      try {
        base.criteria = JSON.parse(critEl.value);
        if (JSON.stringify(base.criteria) !== JSON.stringify(SCHEMA[key].criteria)) edited = true;
      } catch {
        toast(`${key}.criteria 不是合法 JSON，已用默认值`);
      }
    }
    out[key] = base;
  }
  return edited ? out : SCHEMA;
}

async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${url}`);
  const total = Number(res.headers.get("content-length") || 0);
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress?.(received, total);
  }
  const out = new Uint8Array(received);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out;
}

async function loadModel() {
  const bar = $("#load-bar");
  const status = $("#load-status");
  const setStatus = (t) => { status.textContent = t; };
  const setProgress = (f) => { bar.style.width = Math.min(100, f * 100).toFixed(1) + "%"; };

  setStatus("加载推理引擎…");
  setProgress(0.02);
  const ort = await import(ORT_URL);
  ort.env.wasm.numThreads = 1;
  const { AutoTokenizer, env } = await import(TRANSFORMERS_URL);
  // Both enabled: MODEL_BASE is a full URL on HF, a relative path when
  // the weights are self-hosted next to the page.
  env.allowLocalModels = true;
  env.allowRemoteModels = true;
  env.localModelPath = "/";

  setStatus("加载分词器…");
  setProgress(0.04);
  // transformers.js's from_pretrained() cannot load a tokenizer that is a
  // plain file on the same origin: its path parser rejects anything that is
  // not a user/repo model id or a directory with tokenizer.json inside, and
  // its internal fetch fails on the Space's Xet 302 redirects. Load the
  // gzip'd tokenizer.json ourselves (5 MB vs 34 MB) and construct the
  // tokenizer directly from the parsed JSON — verified to work in-browser.
  const [tokCfg, tokJson] = await Promise.all([
    fetch("./tokenizer/tokenizer_config.json").then((r) => r.json()),
    fetch("./tokenizer/tokenizer.json.gz")
      .then((r) => r.arrayBuffer())
      .then((buf) => new Response(new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"))).text())
      .then((text) => JSON.parse(text)),
  ]);
  const tokClsName = (tokCfg.tokenizer_class || "PreTrainedTokenizer").replace(/Fast$/, "");
  const tokCls = AutoTokenizer.TOKENIZER_CLASS_MAPPING[tokClsName] || AutoTokenizer.TOKENIZER_CLASS_MAPPING["PreTrainedTokenizer"];
  const tok = new tokCls(tokJson, tokCfg);
  // The mmBERT tokenizer.json declares
  // PreTokenizer=Metaspace(prepend_scheme="always"), which the Rust tokenizers
  // library honours but transformers.js does not (it warns "Unknown tokenizer
  // class"). Without the prepended "▁", every first token is wrong and benign
  // inputs flip to scam. Prepending one literal space reproduces it — and the
  // normalizer already maps a leading space to "▁", so skip texts that have one.
  const rawEncode = (t) => Array.from(
    tok.encode(t.startsWith(" ") ? t : " " + t, { add_special_tokens: false })
  );

  setStatus("下载模型图…");
  const graph = await fetchWithProgress(`${MODEL_BASE}/model.onnx`);
  setStatus("下载模型权重 (681 MB，仅首次)…");
  const weights = await fetchWithProgress(`${MODEL_BASE}/model.onnx.data`, (r, t) =>
    setProgress(0.05 + 0.9 * (r / t)));

  setStatus("初始化模型…");
  setProgress(0.96);
  const session = await ort.InferenceSession.create(graph, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
    externalData: [{ path: "model.onnx.data", data: weights }],
  });

  setProgress(1);
  setStatus("模型就绪");
  return { ort, session, rawEncode };
}

async function run() {
  const btn = $("#run");
  btn.disabled = true;
  btn.textContent = "分析中…";
  try {
    const text = $("#text").value;
    const questions = editedQuestions();
    const { ort, session, rawEncode } = window.__MODEL;
    const answers = {};
    let totalMs = 0;
    for (const p of selectedPrimitives()) {
      const key = KEY_BY_PRIMITIVE[p];
      const q = questions[key];
      if (!q) continue;
      const { ms, answer } = await predictOne(session, ort, rawEncode, text, q, QTYPE[p]);
      answers[key] = answer;
      totalMs += ms;
    }
    const box = $("#results");
    box.innerHTML = "";
    for (const [key, ans] of Object.entries(answers)) {
      if (RENDERERS[key]) box.insertAdjacentHTML("beforeend", RENDERERS[key](ans));
    }
    $("#meta").textContent =
      `模型: fp16 · 浏览器内推理 · ${totalMs.toFixed(0)} ms · 文本未离开本机`;
  } catch (e) {
    toast("推理失败: " + (e && e.message || e));
  } finally {
    btn.textContent = "开始分析";
    updateRunState();
  }
}

async function init() {
  $("#text").addEventListener("input", updateRunState);
  $$(".primitive").forEach((c) =>
    c.addEventListener("change", () => { updateRunState(); renderAdvanced(); }));
  $("#run").addEventListener("click", run);

  const box = $("#samples");
  for (const s of SAMPLES) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = s.label;
    b.onclick = () => { $("#text").value = s.text; updateRunState(); };
    box.appendChild(b);
  }

  renderAdvanced();
  updateRunState();

  try {
    window.__MODEL = await loadModel();
    window.__MODEL_READY = true;
    $("#load-panel").classList.add("hidden");
    $("#health-badge").textContent = "模型就绪";
    $("#health-badge").className = "badge badge-ok";
  } catch (e) {
    $("#load-status").textContent = "模型加载失败：" + (e && e.message || e);
    $("#health-badge").textContent = "加载失败";
    $("#health-badge").className = "badge badge-err";
  }
  updateRunState();
}

init();