# AGENTS.md

Local ONNX Runtime deployment + LoRA fine-tuning of the Laya decision model for
Chinese/English scam-phrase detection. Inference has **no PyTorch dependency**
(onnxruntime + tokenizers + numpy only). Background: `research/laya-jev-research-report.md`.

## Commands

```bash
# Setup (venv in use is Python 3.14; project requires >=3.10)
source .venv/bin/activate
pip install -e ".[dev]"     # pytest + ruff
pip install -e ".[serve]"   # fastapi/uvicorn/httpx for server & test_server.py

# Tests — run from repo root (tests use relative paths like models/, schemas/)
pytest                       # full suite: 115 passed, 2 skipped, ~30s
pytest tests/test_router.py  # single file
pytest -k route               # by name

# Models (gitignored) — inference and model-dependent tests fail/skip without them
bash scripts/download_models.sh   # English base -> models/laya-onnx-en (curl, bypasses HF proxy issues)
hf download LiChace/laya-scam-detector-onnx --local-dir models/laya-onnx-multilingual-finetuned-fp16

# CLI
python main.py --predict "您好…" --questions schemas/scam.json
python main.py --route "中文字符串"           # no model load needed
python main.py --eval --input datasets/eval.jsonl --output reports/

# Eval acceptance gate (design-doc thresholds: FPR<=0.02, zh acc>=0.90, recall>=0.95, 14-class acc>=0.70, rebate_scam recall>=0.80)
PYTHONPATH=. python scripts/check_acceptance.py --hardneg reports/eval-*.md --holdout ... --handwritten ...

# One-shot v3 pipeline (builds 45k from real CCL + invariants + background training); CCL must be in datasets/raw/ccl2023/
bash scripts/run_v3_pipeline.sh

# Web playground (FastAPI, serves web/ static frontend)
PYTHONPATH=. python -m uvicorn server.app:app --port 8000   # http://127.0.0.1:8000

# Training / export (Apple Silicon MPS; HF_HUB_DISABLE_XET=1 required for HF downloads)
HF_HUB_DISABLE_XET=1 python scripts/train_local_lora.py --train datasets/training/subset30k_train.jsonl ...
HF_HUB_DISABLE_XET=1 python scripts/export_local_onnx.py
```

## Layout

- `main.py` — CLI entrypoint. Imports `from src.router import ...`; the package is
  literally the `src/` directory, so run from repo root or set `PYTHONPATH=.`
- `src/` — inference core. `laya_onnx.py` (`OnnxLayaClient`; question primitives:
  `noul` / `score` / `choice`), `router.py` (`ScriptRouter` unicode-script routing
  → english / multilingual checkpoints), `eval.py`, `report.py`
- `schemas/` — `scam.json` (3-primitive question schema) and `scam_categories.py`
  (canonical 13-class taxonomy + per-dataset label normalization)
- `server/` — FastAPI app; config via env `LAYA_ENGLISH_DIR`, `LAYA_MULTILINGUAL_DIR`,
  `LAYA_HOST`, `LAYA_PORT`. Router loads lazily at startup; `/api/health` reports
  load errors instead of crashing
- `web/` — playground UI served by `server/`. `web-static/` — separate in-browser
  HF Static Space variant (bundles gzipped tokenizer)
- `scripts/` — data prep, LoRA training, ONNX export/quantize, HF pushers (`deploy/`)
- `scripts/dataset_loaders.py` — loaders for 4 new v2 sources (CCL2023-FCC, ChiFraud, TeleAntiFraud-28k, Phishing Email Dataset) with Task 4-discovered real-structure handling (tab-separated ChiFraud, numeric IDs via class.txt, 0/1 phishing labels, etc.)
- `scripts/dataset_mix.py` — `compose_train` (quotas/caps/cross-bucket per-category cap with ccl_nr filtering + final safety assertion), `stable_id` (matches `build_dataset.stable_id` formula)
- `scripts/generate_negatives.py` — 8-genre + nb + contrastive-pair prompt-based generation via `arkcli +chat` (multimodel rotation)
- `scripts/mine_hard_negatives.py` — mining round (mine: score candidate pool → Top-K review list; merge: append verdict=keep rows to train_pool.jsonl)
- `scripts/build_hardneg_evalset.py` — 75/genre carve-out with id-disjoint invariant
- `datasets/hard_negatives/` — 9 genre files (each 600-1200 rows) + contrastive_pairs.jsonl (~5701 rows total); tracked as evidence
- `datasets/hardneg_eval.jsonl` — 675 rows (75x9 genres); never enters training; used by Gate 1
- `models/` — gitignored checkpoints. Code defaults expect `models/laya-onnx-en` +
  `models/laya-onnx-multilingual-finetuned` (variants `-fp16`/`-int8` exist; Docker
  ships fp16 only with English disabled, port 7860). v2 model lands in `models/laya-onnx-multilingual-finetuned-v2/`
- `tests/` — pytest; model-dependent classes `skipif` on missing checkpoints, so the
  suite passes on a fresh clone. Tokenization-quality tests need `TEST_MULTILINGUAL_TOKENIZER=1`
  (downloads from HF). `tests/test_cli.py` strips proxy env vars in subprocesses
- `reports/` — **tracked in git on purpose** ("evaluation reports are evidence")
- `docs/plans/` — design + impl doc pairs per feature (date-prefixed); `kaggle/` is the
  cloud-training fallback path. v2 plan: `docs/plans/2026-09-30-laya-fp-reduction-{design,impl}.md`

## Gotchas

- The `laya-scam` console script in pyproject.toml is **broken** (`ModuleNotFoundError:
  No module named 'main'` — setuptools doesn't package top-level `main.py`). Use `python main.py`.
- `HF_HUB_DISABLE_XET=1` is required for HF downloads (xet backend fails here);
  `download_models.sh` uses curl for the same reason (SOCKS proxy issue with hf CLI).
- `ruff` is in dev deps but has **no config and the repo is not ruff-clean**
  (~89 findings). Don't treat `ruff check` as a gate and don't mass-fix unasked —
  `pytest` is the verification gate. No CI exists.
- Pytest config: `testpaths=["tests"]`, `-v --strict-markers` (unknown marks fail).
- `pytest` is **flaky at exit**: roughly 1 run in 3 aborts with `exit=134` and
  `libc++abi: terminating ... recursive_mutex lock failed` *after* printing the
  summary. All 171 tests still pass; it is a native-destructor race at interpreter
  exit (onnxruntime + torch + the Laya SDK tokenizers loaded in one process). Judge
  the suite by the `N passed` line, not the exit code. Runs with `-p no:cov` or
  `-p no:anyio` happened to avoid it but it is not deterministic.
- Git style: conventional commits (`feat(scope):`, `fix:`, `chore:`).
- `datasets/raw/`, `datasets/training/`, `models/`, `dist/` are gitignored and
  regenerated by scripts; don't commit their contents.
- v2 taxonomy is 14-class (rebate_scam added after job_scam per user ruling); see `schemas/scam.json` and `schemas/scam_categories.py:CCL2023_LABEL_MAP` for the strict 12-class CCL strings (real strings from official README §2.3, NOT provisional)
- **rebate_scam now comes from real CCL2023** (landed at `datasets/raw/ccl2023/{train,test}.json`, copied from the user's `datasets/ccl2023/` which is gitignored). `build_dataset.compose_train` still enforces `rebate_scam >= 25% of positives`; the class is sourced via `load_ccl2023` (28,367 real 刷单返利 rows, 12-class strict map `CCL2023_LABEL_MAP`). The synthetic generator is retired: `datasets/synthetic/rebate_scam.jsonl` was **deleted** and `scripts/gen_rebate_synthetic.py` is no longer wired (kept for history). Note `test.json` is unlabelled (competition test set) — not usable for eval.
- **v3 fixes Gate4 (4/4 gates)**: rebate_scam category recall **0.000 → 0.950** (target ≥0.80) after retraining v3 from scratch on real CCL with the v2 recipe unchanged. Full result `reports/gate_v3_result.md`: FPR 0.006 (≤0.02), holdout 0.989/0.998, handwritten 0.711 (≥0.70), rebate 0.950. Build: `bash scripts/run_v3_pipeline.sh`; same-source holdout: `python scripts/build_ccl_holdout.py` (1000 rebate + 50/label, disjoint by 案件编号, seed 42).
- Gate4 measurement uses **two views**: `datasets/ccl_rebate_eval.jsonl` (CCL same-source holdout, primary, n=1000 rebate) + `datasets/rebate_eval_real.jsonl` (42 real police/media cases, cross-register). v3 scores ~0.95 on both, so the gain generalises past CCL's register. Any rebate eval set must hold **real** narratives — an earlier template version repeated phrases 43x and shared 4-grams with training and measured nothing.
- Eval-set exclusion is by **normalised text**, not id: `stable_id` hashes `source`, and eval rows carry a different source than training rows, so id-only exclusion never fires. See `dataset_mix._norm_text`.
- arkcli data-plane credentials (`~/.arkcli/.env`) expire on the plan timestamp (the Agent Plan Medium tier expires_at was 2026-08-29, with STS refresh_token invalid). When broken: the v2 hard-negative generation fell back to dispatching subagents via `task()` (category quick → arkcodingplan/doubao-seed-2.1-turbo, unspecified-low → doubao-seed-2.0-lite, unspecified-high → kimi-k2.6) as a 3-family interim channel — this produced the 5701-row hard-negatives/ dataset. The `scripts/generate_negatives.py` arkcli path is the documented regeneration route for when credentials are fixed
- v1 baseline on the new hard-negative eval set: hardneg FPR = **59.3%** (job_ad 0.960, promotion 0.933, financial 0.813, propaganda 0.693, personal_social 0.653, ecommerce 0.547, natural_benign 0.467, gov 0.133, traffic_funnel 0.133) — empirically confirms the design thesis: the FP problem is structural and genre-driven
