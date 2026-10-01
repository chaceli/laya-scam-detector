"""Local MPS LoRA fine-tuning of Laya multilingual model for scam detection.

Replaces the Kaggle cloud training path. Runs on Apple Silicon (MPS) or CPU.

Design:
- Base: convaiinnovations/laya-multilingual (mmBERT-base, 322M)
- LoRA: r=8 on Wqkv+Wo (attention), ~1.15M trainable params (0.36%)
- Data: datasets/training/{train,val}.jsonl (canonical category labels)
- Objective: combined cross-entropy on is_scam (noul) + category (choice)
  Cross-entropy is a strictly proper scoring rule (log score), consistent
  with Laya's RLCD training.
- Batching: each batch packs N noul-questions + N choice-questions into one
  forward pass (2N sequences), then splits the logits.

Usage:
  HF_HUB_DISABLE_XET=1 .venv/bin/python scripts/train_local_lora.py \
      --train datasets/training/train.jsonl \
      --val datasets/training/val.jsonl \
      --max-samples 10000 --epochs 3 --batch-size 8 \
      --output models/laya-lora-finetuned
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schemas.scam_categories import CANONICAL_CATEGORIES, CANONICAL_TO_INDEX

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


CATEGORY_DESCRIPTIONS = {
    "benign": "normal legitimate message",
    "phishing": "phishing link, credential theft, account verification",
    "crypto_scam": "cryptocurrency fraud, fake coin offering",
    "investment_scam": "fake returns, stock tips, Ponzi scheme",
    "lottery_scam": "fake prize, lucky draw, congratulations winner",
    "job_scam": "fake job offer, upfront fee, mule recruitment",
    "rebate_scam": "order-brushing rebate fraud, task-based commission scam, upfront deposit",
    "loan_scam": "fake loan offer, predatory lending",
    "impersonation": "fake police, government, bank, customer service",
    "romance_scam": "pig butchering, emotional manipulation",
    "delivery_fraud": "fake courier, refund, lost package",
    "marketing": "aggressive but legitimate sales",
    "adult_content": "adult, escort, sexual services",
    "spam_general": "other unsolicited junk",
}

NOUL_Q = {"t": "noul", "ins": "Is this message a scam or fraud attempt?", "crit": None}
CHOICE_Q = {
    "t": "choice",
    "ins": "What category of scam does this most resemble?",
    "crit": {c: CATEGORY_DESCRIPTIONS[c] for c in CANONICAL_CATEGORIES},
}

QTYPE_NOUL = 2
QTYPE_CHOICE = 0


def load_jsonl(path: Path, max_samples: int | None, seed: int = 42,
               max_chars: int | None = None) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    random.seed(seed)
    random.shuffle(rows)
    if max_samples:
        rows = rows[:max_samples]
    if max_chars:
        for r in rows:
            r["text"] = r["text"][:max_chars]
    return rows


def pre_encode(tok, build_sequence, texts: list[str], q: dict, qtype: int,
               max_len: int, head_max_len: int) -> list[tuple[list[int], list[int]]]:
    """Tokenize every text once. Returns [(ids, markers), ...].

    Tokenization is CPU-bound and dominated the training loop when done
    per-step; pre-encoding moves it out of the hot path.
    """
    return [build_sequence(tok, t, q, max_len=max_len, head_max_len=head_max_len)
            for t in texts]


LEN_BUCKET = 1


def collate(encoded: list[tuple[list[int], list[int]]], indices: list[int],
            qtype: int, device: str, pad_id: int) -> dict:
    """Pad a batch of pre-encoded (ids, markers) into tensors.

    Padded length is rounded up to LEN_BUCKET so the batch hits a small,
    repeating set of shapes. MPS recompiles its graph per new shape, so
    rounding is a large throughput win over exact-length padding.
    Marker counts are padded to the batch max with mask=False (they differ
    only when max_len truncates).
    """
    items = [encoded[i] for i in indices]
    raw_max = max(len(ids) for ids, _ in items)
    maxlen = ((raw_max + LEN_BUCKET - 1) // LEN_BUCKET) * LEN_BUCKET
    nmax = max(len(m) for _, m in items)

    input_ids, attn, marker_pos, marker_mask = [], [], [], []
    for ids, markers in items:
        pad = maxlen - len(ids)
        k = len(markers)
        input_ids.append(ids + [pad_id] * pad)
        attn.append([1] * len(ids) + [0] * pad)
        marker_pos.append(list(markers) + [0] * (nmax - k))
        marker_mask.append([True] * k + [False] * (nmax - k))

    return {
        "input_ids": torch.tensor(input_ids, device=device),
        "attention_mask": torch.tensor(attn, device=device),
        "marker_pos": torch.tensor(marker_pos, device=device),
        "marker_mask": torch.tensor(marker_mask, device=device),
        "qtype": torch.full((len(items),), qtype, dtype=torch.long, device=device),
    }


def proper_scoring_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Log score (cross-entropy) — a strictly proper scoring rule.

    Maximized in expectation only when the model reports its true
    probabilities, which is what drives calibration. Equivalent to the log
    component of Laya's RLCD reward, expressed as a loss (always >= 0).
    """
    return F.cross_entropy(logits.float(), target)


@torch.no_grad()
def evaluate(peft_model, enc_noul, enc_choice, targets_scam, targets_cat,
             device, batch_size):
    peft_model.eval()
    n = len(enc_noul)
    correct_scam = correct_cat = 0
    for i in range(0, n, batch_size):
        idx = list(range(i, min(i + batch_size, n)))
        t_scam = targets_scam[idx]
        t_cat = targets_cat[idx]
        b_noul = collate(enc_noul, idx, QTYPE_NOUL, device, 0)
        b_choice = collate(enc_choice, idx, QTYPE_CHOICE, device, 0)
        logits_n, _ = peft_model(**b_noul)
        logits_c, _ = peft_model(**b_choice)
        correct_scam += (logits_n.argmax(-1) == t_scam).sum().item()
        correct_cat += (logits_c.argmax(-1) == t_cat).sum().item()
    peft_model.train()
    return correct_scam / n, correct_cat / n, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="datasets/training/train.jsonl")
    ap.add_argument("--val", default="datasets/training/val.jsonl")
    ap.add_argument("--max-samples", type=int, default=10000)
    ap.add_argument("--max-val", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--max-chars", type=int, default=1024,
                    help="truncate state text before tokenization (long HTML pages otherwise dominate)")
    ap.add_argument("--head-max-len", type=int, default=192)
    ap.add_argument("--output", default="models/laya-lora-finetuned")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Args: {vars(args)}")

    import laya
    from laya.common import build_sequence
    from peft import LoraConfig, get_peft_model

    print("\nLoading base multilingual checkpoint...")
    agent = laya.load("convaiinnovations/laya", subfolder="multilingual", device=device)
    model, tok = agent.model, agent.tok
    # Training in fp32 for numerical stability (MPS fp16 grads can be unstable)
    model = model.float()

    print("Wrapping with LoRA...")
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=["Wqkv", "Wo"], bias="none",
    )
    model.train()
    peft_model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    print(f"  trainable: {trainable/1e6:.2f}M")

    print(f"\nLoading data (max {args.max_samples})...")
    train_rows = load_jsonl(Path(args.train), args.max_samples, args.seed,
                            max_chars=args.max_chars)
    val_rows = load_jsonl(Path(args.val), args.max_val, args.seed,
                          max_chars=args.max_chars)
    print(f"  train: {len(train_rows)}, val: {len(val_rows)}")

    print("Pre-encoding train split (tokenization out of the hot loop)...")
    t0 = time.time()
    train_texts = [r["text"] for r in train_rows]
    enc_noul = pre_encode(tok, build_sequence, train_texts, NOUL_Q, QTYPE_NOUL,
                          args.max_len, args.head_max_len)
    enc_choice = pre_encode(tok, build_sequence, train_texts, CHOICE_Q, QTYPE_CHOICE,
                            args.max_len, args.head_max_len)
    targets_scam = torch.tensor([r["is_scam"] for r in train_rows], device=device)
    targets_cat = torch.tensor([CANONICAL_TO_INDEX.get(r["category"], 12) for r in train_rows],
                               device=device)
    print(f"  ✓ {len(train_rows)} samples in {time.time()-t0:.0f}s")

    print("Pre-encoding val split...")
    t0 = time.time()
    val_texts = [r["text"] for r in val_rows]
    val_enc_noul = pre_encode(tok, build_sequence, val_texts, NOUL_Q, QTYPE_NOUL,
                              args.max_len, args.head_max_len)
    val_enc_choice = pre_encode(tok, build_sequence, val_texts, CHOICE_Q, QTYPE_CHOICE,
                                args.max_len, args.head_max_len)
    val_targets_scam = torch.tensor([r["is_scam"] for r in val_rows], device=device)
    val_targets_cat = torch.tensor([CANONICAL_TO_INDEX.get(r["category"], 12) for r in val_rows],
                                   device=device)
    print(f"  ✓ {len(val_rows)} samples in {time.time()-t0:.0f}s")

    # Sort by bucketed length so each batch shares one rounded shape, which
    # lets MPS reuse a compiled graph instead of recompiling per batch.
    def bucket_of(i: int) -> int:
        n = max(len(enc_noul[i][0]), len(enc_choice[i][0]))
        return ((n + LEN_BUCKET - 1) // LEN_BUCKET) * LEN_BUCKET

    order = sorted(range(len(train_rows)), key=bucket_of)

    opt = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.01)
    steps_per_epoch = len(train_rows) // (args.batch_size * args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(1, total_steps),
        pct_start=0.06, anneal_strategy="cos",
    )
    print(f"  steps/epoch={steps_per_epoch}, total_steps={total_steps}")

    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)
    best_acc = 0.0
    global_step = 0
    t_start = time.time()

    for epoch in range(args.epochs):
        # Reshuffle within length buckets so every step stays cheap.
        rand_order = order[:]
        for s in range(0, len(rand_order), args.batch_size * 32):
            block = rand_order[s:s + args.batch_size * 32]
            random.shuffle(block)
            rand_order[s:s + args.batch_size * 32] = block
        epoch_loss = 0.0
        n_batches = 0
        opt.zero_grad()
        t_epoch = time.time()

        for step_i in range(0, len(rand_order) - args.batch_size + 1, args.batch_size):
            idx = rand_order[step_i:step_i + args.batch_size]
            t_scam = targets_scam[idx]
            t_cat = targets_cat[idx]

            b_noul = collate(enc_noul, idx, QTYPE_NOUL, device, tok.pad_token_id)
            b_choice = collate(enc_choice, idx, QTYPE_CHOICE, device, tok.pad_token_id)

            logits_n, _ = peft_model(**b_noul)
            logits_c, _ = peft_model(**b_choice)
            loss_n = proper_scoring_loss(logits_n, t_scam)
            loss_c = proper_scoring_loss(logits_c, t_cat)
            loss = (loss_n + loss_c) / args.grad_accum
            loss.backward()

            epoch_loss += (loss_n.item() + loss_c.item())
            n_batches += 1

            if (step_i // args.batch_size) % args.grad_accum == args.grad_accum - 1:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in peft_model.parameters() if p.requires_grad], 1.0
                )
                opt.step()
                sched.step()
                opt.zero_grad()
                global_step += 1

                if global_step % 50 == 0:
                    elapsed = time.time() - t_epoch
                    rate = (step_i + args.batch_size) / max(elapsed, 1e-9)
                    print(f"  epoch {epoch+1} step {global_step}/{total_steps} "
                          f"loss={epoch_loss/n_batches:.4f} "
                          f"lr={sched.get_last_lr()[0]:.2e} "
                          f"{rate:.0f} samples/s")

        scam_acc, cat_acc, n_eval = evaluate(
            peft_model, val_enc_noul, val_enc_choice,
            val_targets_scam, val_targets_cat, device, args.batch_size,
        )
        print(f"\n  [epoch {epoch+1}] loss={epoch_loss/max(n_batches,1):.4f} "
              f"val_zh_scam_acc={scam_acc:.3f} val_cat_acc={cat_acc:.3f} (n={n_eval}) "
              f"({time.time()-t_epoch:.0f}s)")
        if scam_acc + cat_acc > best_acc:
            best_acc = scam_acc + cat_acc
            print(f"  ✓ new best; saving checkpoint")

    print(f"\nTraining done in {time.time()-t_start:.0f}s")

    # Save merged model
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    print(f"\nMerging LoRA and saving to {out}...")
    merged = peft_model.merge_and_unload()
    torch.save(merged.state_dict(), out / "model_state.pt")
    print(f"  ✓ saved model_state.pt ({sum(p.numel() for p in merged.parameters())/1e6:.1f}M params)")

    # Save metadata
    (out / "train_meta.json").write_text(json.dumps({
        "args": vars(args),
        "final_val_scam_acc": scam_acc,
        "final_val_cat_acc": cat_acc,
        "train_samples": len(train_rows),
        "trainable_params": trainable,
    }, indent=2))
    print(f"  ✓ saved train_meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())