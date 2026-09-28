"""ONNX Runtime wrapper for Laya decision model.

Public API:
    OnnxLayaClient(checkpoint_dir) -> predict(state, questions) -> dict
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


QTYPE_CHOICE = 0
QTYPE_SCORE = 1
QTYPE_NOUL = 2

_QTYPE_FROM_STR = {"choice": QTYPE_CHOICE, "score": QTYPE_SCORE, "noul": QTYPE_NOUL}


def _softmax_with_temperature(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Convert logits to probabilities with temperature scaling."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    scaled = logits.astype(np.float64) / temperature
    scaled = scaled - scaled.max()
    exp = np.exp(scaled)
    return (exp / exp.sum()).astype(np.float32)


def _tokenize_question(
    *,
    tokenizer,
    question: str,
    options: dict[str, str],
    head_max_len: int,
    cls_id: int,
    sep_id: int,
    mask_id: int,
) -> tuple[list[int], list[int]]:
    """Build Laya ONNX question prefix with one [MASK] per option.

    Returns (token_ids, option_positions_in_sequence).
    """
    q_ids = tokenizer.encode("choice question: " + question, add_special_tokens=False).ids

    token_ids: list[int] = [cls_id] + q_ids + [sep_id]
    option_positions: list[int] = []

    per_option_budget = max(8, (head_max_len - 16) // max(len(options), 1))

    for label, description in options.items():
        option_positions.append(len(token_ids))
        token_ids.append(mask_id)
        opt_text = f" {label}: {description}"
        opt_ids = tokenizer.encode(opt_text, add_special_tokens=False).ids
        token_ids.extend(opt_ids[:per_option_budget])
    token_ids.append(sep_id)

    return token_ids, option_positions


def _format_for_noul() -> tuple[dict[str, str], tuple[str, str]]:
    """noul uses exactly 2 options: false, true."""
    options = {"false": "no", "true": "yes"}
    return options, ("false", "true")


def _format_for_score(criteria: list) -> dict[str, str]:
    """score uses levels 0..N-1, presented as level_0..level_N-1."""
    return {f"level_{i}": str(c) for i, c in enumerate(criteria)}


class OnnxLayaClient:
    """Laya decision model client using ONNX Runtime (no PyTorch)."""

    def __init__(
        self,
        checkpoint_dir: Path | str,
        providers: Optional[list[str]] = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.session = ort.InferenceSession(
            str(self.checkpoint_dir / "model.onnx"),
            providers=providers or ["CPUExecutionProvider"],
        )
        self.tokenizer = Tokenizer.from_file(
            str(self.checkpoint_dir / "tokenizer" / "tokenizer.json")
        )
        cfg_path = self.checkpoint_dir / "rl_agent_config.json"
        self.config = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        self.cls_id, self.sep_id, self.mask_id, self.pad_id = self._resolve_special_tokens()
        self.default_max_len = int(self.config.get("max_len", 512))
        self.default_head_max_len = int(self.config.get("head_max_len", 192))

    def _resolve_special_tokens(self) -> tuple[int, int, int, int]:
        """Resolve special token ids across both checkpoint families.

        The English ONNX export uses BERT-style names ([CLS]/[SEP]/[MASK]).
        mmBERT (multilingual) uses <bos>/<eos>/<mask>. Prefer the literal
        names, then fall back to tokenizer_config.json, then to the tokenizer's
        own special-token slots.
        """
        def by_name(name: str) -> Optional[int]:
            return self.tokenizer.token_to_id(name) if name else None

        tok_cfg_path = self.checkpoint_dir / "tokenizer" / "tokenizer_config.json"
        tok_cfg = json.loads(tok_cfg_path.read_text()) if tok_cfg_path.exists() else {}

        def pick(explicit: str, cfg_key: str) -> Optional[int]:
            return by_name(explicit) or by_name(tok_cfg.get(cfg_key, ""))

        cls_id = pick("[CLS]", "cls_token")
        sep_id = pick("[SEP]", "sep_token")
        mask_id = pick("[MASK]", "mask_token")
        pad_id = pick("[PAD]", "pad_token")
        if pad_id is None:
            pad_id = 0
        if any(t is None for t in (cls_id, sep_id, mask_id)):
            raise ValueError(
                f"Tokenizer missing special tokens; looked for [CLS]/[SEP]/[MASK] "
                f"and tokenizer_config.json keys. Got cls={cls_id} sep={sep_id} mask={mask_id}"
            )
        return cls_id, sep_id, mask_id, pad_id

    def _build_input(
        self, state: str, question_text: str, options: dict[str, str],
        max_len: int, head_max_len: int,
    ) -> tuple[list[int], list[int]]:
        """Build (token_ids, option_positions) for state + question prefix."""
        head_tokens, head_positions = _tokenize_question(
            tokenizer=self.tokenizer,
            question=question_text,
            options=options,
            head_max_len=head_max_len,
            cls_id=self.cls_id,
            sep_id=self.sep_id,
            mask_id=self.mask_id,
        )
        if len(head_tokens) >= max_len:
            return head_tokens[:max_len], [
                p for p in head_positions if p < max_len
            ]
        remaining = max_len - len(head_tokens) - 1
        state_ids = self.tokenizer.encode(state, add_special_tokens=False).ids
        state_ids = state_ids[: max(0, remaining)]
        token_ids = head_tokens + state_ids + [self.sep_id]
        option_positions = [p for p in head_positions if p < len(token_ids)]
        return token_ids, option_positions

    def _run_one(
        self, token_ids: list[int], option_positions: list[int], qtype: int,
    ) -> np.ndarray:
        if not option_positions:
            raise ValueError("no options to score")
        seq_len = len(token_ids)
        n_opts = len(option_positions)
        inputs = {
            "input_ids": np.array([token_ids], dtype=np.int64),
            "attention_mask": np.ones((1, seq_len), dtype=np.int64),
            "marker_pos": np.array([option_positions + [0] * (16 - n_opts)], dtype=np.int64)[:, :n_opts],
            "marker_mask": np.zeros((1, 16), dtype=bool),
            "qtype": np.array([qtype], dtype=np.int64),
        }
        # Set valid positions
        mask = np.zeros((1, 16), dtype=bool)
        mask[0, :n_opts] = True
        inputs["marker_mask"] = mask
        # Pad marker_pos to fixed width (model expects fixed-size)
        padded = np.zeros((1, 16), dtype=np.int64)
        padded[0, :n_opts] = option_positions
        inputs["marker_pos"] = padded

        outputs = self.session.run(None, inputs)
        return outputs[0][0, :n_opts]

    def _temperature_for(self, qtype: int, option_count: int) -> float:
        """Look up fitted temperature from rl_agent_config.

        Schema:
        - temperature_by_options: dict with keys like "choice:3-5", "noul:2"
        - temperature: array (used by qtype index: 0=choice, 1=score, 2=noul)
        """
        temps_by_options = self.config.get("temperature_by_options", {})
        qtype_name = {0: "choice", 1: "score", 2: "noul"}[qtype]
        # Exact option count match
        for key, val in temps_by_options.items():
            if key.startswith(f"{qtype_name}:"):
                suffix = key.split(":")[1]
                if "-" in suffix:
                    lo, hi = suffix.split("-")
                    if int(lo) <= option_count <= int(hi):
                        return float(val)
                elif suffix.endswith("+"):
                    if option_count >= int(suffix[:-1]):
                        return float(val)
                elif suffix.isdigit() and option_count == int(suffix):
                    return float(val)
        # Fallback: temperature array indexed by qtype
        temps_arr = self.config.get("temperature", [])
        if qtype < len(temps_arr):
            return float(temps_arr[qtype])
        return 1.0

    def _answer_for_qtype(
        self, qtype: int, logits: np.ndarray, criteria: list | dict,
        temperature: float,
    ) -> dict:
        probs = _softmax_with_temperature(logits, temperature)
        if qtype == QTYPE_NOUL:
            # probs[0] = P(false), probs[1] = P(true)
            p_true = float(probs[1]) if len(probs) > 1 else float(probs[0])
            return {
                "noul": p_true,
                "probabilities": {"false": float(probs[0]), "true": p_true},
                "confidence": max(float(probs[0]), p_true),
            }
        if qtype == QTYPE_CHOICE:
            labels = list(criteria.keys())
            dist = {labels[i]: float(probs[i]) for i in range(len(labels))}
            argmax_idx = int(np.argmax(probs))
            return {
                "choice": labels[argmax_idx],
                "probabilities": dist,
                "confidence": float(probs[argmax_idx]),
            }
        if qtype == QTYPE_SCORE:
            n = len(probs)
            expected = sum(float(probs[i]) * i for i in range(n))
            levels = criteria if isinstance(criteria, list) else list(criteria.values())
            dist = {levels[i]: float(probs[i]) for i in range(n)}
            return {
                "score": expected,
                "max_score": n - 1,
                "distribution": dist,
                "confidence": float(probs.max()),
            }
        raise ValueError(f"unknown qtype: {qtype}")

    def predict(
        self,
        state: str,
        questions: dict,
        max_len: Optional[int] = None,
        head_max_len: Optional[int] = None,
    ) -> dict:
        import time
        max_len = max_len or self.default_max_len
        head_max_len = head_max_len or self.default_head_max_len

        if not isinstance(state, str) or not state:
            raise ValueError("state must be a non-empty string")

        answers: dict = {}
        started = time.perf_counter()
        for name, q in questions.items():
            qtype_str = q.get("type", "noul")
            if qtype_str not in _QTYPE_FROM_STR:
                raise ValueError(f"unknown question type: {qtype_str}")
            qtype = _QTYPE_FROM_STR[qtype_str]
            instructions = q.get("instructions", "")
            criteria = q.get("criteria", [])

            if qtype == QTYPE_CHOICE:
                if not isinstance(criteria, dict) or len(criteria) < 2:
                    raise ValueError(f"choice question '{name}' needs criteria dict with >=2 keys")
                options = {k: str(v) for k, v in criteria.items()}
            elif qtype == QTYPE_NOUL:
                options, _ = _format_for_noul()
            elif qtype == QTYPE_SCORE:
                if not isinstance(criteria, list) or len(criteria) < 2:
                    raise ValueError(f"score question '{name}' needs criteria list with >=2 levels")
                options = _format_for_score(criteria)

            tokens, positions = self._build_input(
                state=state,
                question_text=instructions,
                options=options,
                max_len=max_len,
                head_max_len=head_max_len,
            )
            logits = self._run_one(tokens, positions, qtype)
            temperature = self._temperature_for(qtype, len(options))
            answers[name] = self._answer_for_qtype(qtype, logits, criteria, temperature)
        latency_ms = (time.perf_counter() - started) * 1000
        return {
            "answers": answers,
            "routing": {"model": "unknown", "reason": "OnnxLayaClient direct call"},
            "latency_ms": latency_ms,
        }

    def predict_batch(self, requests: list[dict]) -> list[dict]:
        return [self.predict(**r) for r in requests]