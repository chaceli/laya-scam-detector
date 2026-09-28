"""Export a locally-trained Laya model (model_state.pt) to ONNX.

Complements scripts/merge_and_export_onnx.py (the HF-adapter path) for the
local MPS training flow, where scripts/train_local_lora.py writes:
    models/laya-lora-finetuned/model_state.pt   (merged DecisionModel)

Output matches the OnnxLayaClient contract in src/laya_onnx.py:
    inputs:  input_ids, attention_mask, marker_pos, marker_mask, qtype
    output:  logits

Usage:
  HF_HUB_DISABLE_XET=1 .venv/bin/python scripts/export_local_onnx.py \
      --state models/laya-lora-finetuned/model_state.pt \
      --output models/laya-onnx-multilingual-finetuned
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schemas.scam_categories import CANONICAL_CATEGORIES


class OnnxWrapper(torch.nn.Module):
    """Drop detach_encoder and keep the 5-input / 1-output ONNX contract."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        logits, _ = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            marker_pos=marker_pos,
            marker_mask=marker_mask,
            qtype=qtype,
        )
        return logits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="models/laya-lora-finetuned/model_state.pt")
    ap.add_argument("--output", default="models/laya-onnx-multilingual-finetuned")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--n-opts", type=int, default=16)
    ap.add_argument("--verify", action="store_true", default=True,
                    help="compare PyTorch vs ONNX logits on a sample input")
    args = ap.parse_args()

    device = "cpu"
    print("Loading base multilingual model via Laya SDK...")
    import laya
    agent = laya.load("convaiinnovations/laya", subfolder="multilingual", device=device)
    model, tok = agent.model, agent.tok
    model = model.float()

    state_path = Path(args.state)
    print(f"Loading trained state dict from {state_path}...")
    state = torch.load(state_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  ⚠ {len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        print(f"  ⚠ {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
    print("  ✓ state loaded")
    model.eval()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"

    wrapper = OnnxWrapper(model).eval()
    dummy = (
        torch.zeros((1, args.seq_len), dtype=torch.long),
        torch.ones((1, args.seq_len), dtype=torch.long),
        torch.zeros((1, args.n_opts), dtype=torch.long),
        torch.ones((1, args.n_opts), dtype=torch.bool),
        torch.zeros((1,), dtype=torch.long),
    )
    print(f"\nExporting ONNX → {onnx_path}")
    torch.onnx.export(
        wrapper,
        dummy,
        str(onnx_path),
        input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "marker_pos": {0: "batch", 1: "n_opts"},
            "marker_mask": {0: "batch", 1: "n_opts"},
            "logits": {0: "batch", 1: "n_opts"},
        },
        opset_version=14,
    )
    print(f"  ✓ {onnx_path} ({onnx_path.stat().st_size:,} bytes)")

    # Copy tokenizer + config
    tok_dir = out_dir / "tokenizer"
    tok_dir.mkdir(exist_ok=True)
    tok.save_pretrained(str(tok_dir))
    cfg_src = Path(agent.model_id) if Path(agent.model_id).exists() else None
    print(f"  ✓ tokenizer → {tok_dir}")

    (out_dir / "categories.json").write_text(
        json.dumps({"categories": CANONICAL_CATEGORIES}, ensure_ascii=False, indent=2)
    )
    print("  ✓ categories.json (13 classes)")

    if args.verify:
        print("\nVerifying PyTorch vs ONNX...")
        import onnxruntime as ort
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        probe = (
            "您好，我是XX快递客服，您有一个包裹在运输途中丢失，请点击链接理赔。",
        )
        from laya.common import build_sequence
        q = {"t": "noul", "ins": "Is this a scam?", "crit": None}
        ids, markers = build_sequence(tok, probe[0], q, max_len=args.seq_len,
                                      head_max_len=192)
        n = len(markers)
        pos = markers + [0] * (args.n_opts - n)
        mask = [True] * n + [False] * (args.n_opts - n)
        feed = {
            "input_ids": np.array([ids], dtype=np.int64),
            "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
            "marker_pos": np.array([pos], dtype=np.int64),
            "marker_mask": np.array([mask], dtype=bool),
            "qtype": np.array([2], dtype=np.int64),
        }
        onnx_logits = session.run(None, feed)[0][0][:n]
        with torch.no_grad():
            pt_logits, _ = model(
                input_ids=torch.tensor([ids]),
                attention_mask=torch.ones((1, len(ids)), dtype=torch.long),
                marker_pos=torch.tensor([markers]),
                marker_mask=torch.tensor([[True] * n]),
                qtype=torch.tensor([2]),
            )
        pt = pt_logits[0][:n].numpy()
        diff = float(np.max(np.abs(pt - onnx_logits)))
        print(f"  PyTorch: {pt.tolist()}")
        print(f"  ONNX:    {onnx_logits.tolist()}")
        print(f"  max diff: {diff:.2e}")
        if diff > 1e-3:
            print("  ⚠ logit diff above 1e-3 — check export")
            return 1
        print("  ✓ ONNX matches PyTorch")

    print(f"\n✓ Done. Bundle at {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())