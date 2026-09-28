"""Benchmark MPS LoRA training throughput to calibrate training size.

Run:
  HF_HUB_DISABLE_XET=1 .venv/bin/python scripts/bench_mps_throughput.py
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
    import laya
    from laya.common import build_sequence
    from peft import LoraConfig, get_peft_model

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    agent = laya.load("convaiinnovations/laya", subfolder="multilingual", device=device)
    model, tok = agent.model, agent.tok

    lora_cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                          target_modules=["Wqkv", "Wo"], bias="none")
    model.train()
    peft_model = get_peft_model(model, lora_cfg)
    opt = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad], lr=2e-4)

    # Build a pool of training sequences
    texts = [
        "您好，我是XX快递客服，您有一个包裹在运输途中丢失，请点击链接理赔。",
        "【最高人民检察院】您涉嫌洗钱，请将资金转入安全账户配合调查。",
        "恭喜您被抽中二等奖，奖金10万元，请尽快联系客服领取。",
        "妈，我今晚回家吃饭，大概6点到家。",
        "老板，明天上午9点的会议改到下午3点，请准时参加。",
        "您的快递已签收，感谢您选择京东，期待再次为您服务。",
        "Make $5000/day with our secret crypto trading algorithm. Join now!",
        "Hi, just confirming our meeting at 3pm tomorrow.",
    ]
    q_noul = {"t": "noul", "ins": "Is this a scam?", "crit": None}
    seqs = [build_sequence(tok, t, q_noul, max_len=256, head_max_len=192)[0] for t in texts]
    print(f"Sequence lengths: {[len(s) for s in seqs]}")

    def make_batch(batch_seqs):
        maxlen = max(len(s) for s in batch_seqs)
        input_ids, attn, marker_pos, marker_mask = [], [], [], []
        for s in batch_seqs:
            pad = maxlen - len(s)
            input_ids.append(s + [tok.pad_token_id] * pad)
            attn.append([1] * len(s) + [0] * pad)
            marker_pos.append([0] + [0] * 15)  # single marker
            marker_mask.append([True] + [False] * 15)
        return {
            "input_ids": torch.tensor(input_ids, device=device),
            "attention_mask": torch.tensor(attn, device=device),
            "marker_pos": torch.tensor(marker_pos, device=device),
            "marker_mask": torch.tensor(marker_mask, device=device),
            "qtype": torch.full((len(batch_seqs),), 2, dtype=torch.long, device=device),
        }

    print("\nBenchmarking (5 steps per config):")
    for bs in [4, 8, 16]:
        batch = make_batch([seqs[i % len(seqs)] for i in range(bs)])
        target = torch.randint(0, 2, (bs,), device=device)
        # Warmup
        out, _ = peft_model(**batch)
        loss = F.cross_entropy(out.float(), target)
        loss.backward()
        opt.step()
        opt.zero_grad()
        if device == "mps":
            torch.mps.synchronize()

        t0 = time.time()
        n = 5
        for _ in range(n):
            out, _ = peft_model(**batch)
            loss = F.cross_entropy(out.float(), target)
            loss.backward()
            opt.step()
            opt.zero_grad()
        if device == "mps":
            torch.mps.synchronize()
        dt = time.time() - t0
        per_step = dt / n
        samples_per_sec = bs / per_step
        mem = torch.mps.current_allocated_memory() / 1e9 if device == "mps" else 0
        print(f"  bs={bs:2d}: {per_step*1000:6.0f} ms/step, {samples_per_sec:6.1f} samples/s, {mem:.2f} GB")

    print("\nProjection for 30k samples:")
    # use bs=8 rate
    batch = make_batch([seqs[i % len(seqs)] for i in range(8)])
    target = torch.randint(0, 2, (8,), device=device)
    if device == "mps":
        torch.mps.synchronize()
    t0 = time.time()
    for _ in range(5):
        out, _ = peft_model(**batch)
        loss = F.cross_entropy(out.float(), target)
        loss.backward(); opt.step(); opt.zero_grad()
    if device == "mps":
        torch.mps.synchronize()
    rate = 8 * 5 / (time.time() - t0)
    for n_samples in [10000, 30000]:
        for epochs in [2, 3]:
            steps = n_samples / 8
            secs = steps * epochs / (rate / 8)
            print(f"  {n_samples} samples x {epochs} epochs ≈ {secs/60:.0f} min ({secs/3600:.1f}h) at {rate:.1f} samples/s")

    return 0


if __name__ == "__main__":
    sys.exit(main())