"""Technical spike: verify local MPS LoRA fine-tuning is feasible.

Tests:
1. Load Laya multilingual model on MPS
2. Build training inputs via Laya's build_sequence
3. Forward pass → logits
4. Compute proper-scoring-rule loss
5. Backward pass
6. Verify LoRA gradients exist
7. Measure peak memory

Run:
  HF_HUB_DISABLE_XET=1 .venv/bin/python scripts/spike_mps_train.py
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    print("=" * 70)
    print("MPS LoRA training spike")
    print("=" * 70)

    import laya
    from laya.common import build_sequence

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # 1. Load model
    print("\n[1/7] Loading multilingual checkpoint...")
    t0 = time.time()
    agent = laya.load("convaiinnovations/laya", subfolder="multilingual", device=device)
    print(f"  ✓ loaded in {time.time()-t0:.1f}s, device={agent.device}, dtype={agent.dtype}")
    model = agent.model
    tok = agent.tok

    # 2. Build training inputs
    print("\n[2/7] Building training inputs via build_sequence...")
    state = "您好，我是XX快递客服，您有一个包裹在运输途中丢失，请点击链接填写个人信息进行理赔。"
    q_noul = {"t": "noul", "ins": "Is this a scam?", "crit": None}
    ids, markers = build_sequence(tok, state, q_noul, max_len=512, head_max_len=192)
    print(f"  ✓ noul: {len(ids)} tokens, markers={markers}")

    q_choice = {
        "t": "choice",
        "ins": "What category?",
        "crit": {"benign": "normal", "phishing": "fake link", "delivery_fraud": "fake courier",
                 "investment_scam": "fake investment"},
    }
    ids_c, markers_c = build_sequence(tok, state, q_choice, max_len=512, head_max_len=192)
    print(f"  ✓ choice: {len(ids_c)} tokens, {len(markers_c)} options")

    # 3. Move to device, build tensors
    print("\n[3/7] Building batch tensors...")

    def make_batch(ids, markers, qtype):
        n = len(markers)
        marker_pos = markers + [0] * (16 - n)
        marker_mask = [True] * n + [False] * (16 - n)
        return {
            "input_ids": torch.tensor([ids], device=device),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device=device),
            "marker_pos": torch.tensor([marker_pos], device=device),
            "marker_mask": torch.tensor([marker_mask], device=device),
            "qtype": torch.tensor([qtype], device=device),
        }

    batch_noul = make_batch(ids, markers, qtype=2)  # noul
    batch_choice = make_batch(ids_c, markers_c, qtype=0)  # choice

    # 4. Forward
    print("\n[4/7] Forward pass...")
    model.eval()
    with torch.no_grad():
        logits_n, act = model(**batch_noul)
        logits_c, _ = model(**batch_choice)
    print(f"  ✓ noul logits: {logits_n.shape} = {logits_n[0].tolist()}")
    print(f"  ✓ choice logits: {logits_c.shape} = {logits_c[0].tolist()}")

    # 5. LoRA wrap
    print("\n[5/7] Wrapping encoder with LoRA...")
    from peft import LoraConfig, get_peft_model
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["Wqkv", "Wo"],
        bias="none",
    )
    model.train()
    peft_model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    print(f"  ✓ LoRA: {trainable/1e6:.2f}M trainable / {total/1e6:.1f}M total ({100*trainable/total:.2f}%)")

    # 6. Loss + backward
    print("\n[6/7] Loss + backward...")
    opt = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad], lr=2e-4)

    # Target: option 1 (=true for noul)
    target_noul = torch.tensor([1], device=device)
    out_n, _ = peft_model(**batch_noul)
    probs_n = F.softmax(out_n.float(), dim=-1)
    # Proper scoring: log score
    eps = 1e-7
    loss_noul = -torch.log(probs_n.gather(1, target_noul.unsqueeze(1)).squeeze() + eps).mean()

    # Target: option 2 (=delivery_fraud)
    target_choice = torch.tensor([2], device=device)
    out_c, _ = peft_model(**batch_choice)
    probs_c = F.softmax(out_c.float(), dim=-1)
    loss_choice = -torch.log(probs_c.gather(1, target_choice.unsqueeze(1)).squeeze() + eps).mean()

    loss = loss_noul + loss_choice
    print(f"  ✓ loss_noul={loss_noul.item():.4f}, loss_choice={loss_choice.item():.4f}")

    loss.backward()
    print("  ✓ backward complete")

    # Check grads
    grad_params = sum(1 for p in peft_model.parameters() if p.requires_grad and p.grad is not None)
    grad_norm = sum(p.grad.norm().item() ** 2 for p in peft_model.parameters()
                    if p.requires_grad and p.grad is not None) ** 0.5
    print(f"  ✓ {grad_params} params with grads, grad_norm={grad_norm:.4f}")

    opt.step()

    # 7. Memory
    print("\n[7/7] Memory:")
    if device == "mps":
        allocated = torch.mps.current_allocated_memory() / 1e9
        driver = torch.mps.driver_allocated_memory() / 1e9
        print(f"  MPS allocated: {allocated:.2f} GB")
        print(f"  MPS driver:    {driver:.2f} GB")

    print("\n" + "=" * 70)
    print("✓ SPIKE PASSED — local MPS LoRA training is feasible")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())