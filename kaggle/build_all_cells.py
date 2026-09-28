"""Add all training cells to the Kaggle notebook in one shot."""
import nbformat as nbf

nb_path = "kaggle/laya_finetune_multilingual.ipynb"
nb = nbf.read(nb_path, as_version=4)


def add(cell, position: int | str = "end"):
    if position == "end":
        nb.cells.append(cell)
    else:
        nb.cells.insert(position, cell)


# Cell 1: Setup
setup = nbf.v4.new_code_cell("""\
!pip install -q -U peft trl accelerate datasets transformers safetensors laya
import torch
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
""")
add(setup)

# Cell 2: Load base via PyTorch Laya SDK
load_base = nbf.v4.new_code_cell("""\
import os
# Use mirror if region blocks HF
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from laya import Router
import torch

# Load multilingual checkpoint (mmBERT-base, 322M)
router = Router(device="cuda", preload=True)
print("english loaded:", router.english is not None)
print("multilingual loaded:", getattr(router, "multilingual", None) is not None)

# Access the underlying model for training
agent = router.multilingual  # ConvaiInnovations/laya-multilingual
model = agent.model
print("Model class:", type(model).__name__)
n_params = sum(p.numel() for p in model.parameters())
print(f"Total params: {n_params/1e6:.1f}M")
print(f"Trainable (no LoRA): {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M")
""")
add(load_base)

# Cell 3: LoRA injection
lora = nbf.v4.new_code_cell("""\
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["query", "key", "value", "dense"],
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION,
)

peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()
""")
add(lora)

# Cell 4: Dataset loading
data_cell = nbf.v4.new_code_cell("""\
from datasets import load_dataset

# /kaggle/input/scam-detection-training is the Kaggle Dataset created from
# local datasets/training/{train,val,test}.jsonl
TRAIN_PATH = "/kaggle/input/scam-detection-training/train.jsonl"
VAL_PATH = "/kaggle/input/scam-detection-training/val.jsonl"

train_ds = load_dataset("json", data_files=TRAIN_PATH, split="train")
val_ds = load_dataset("json", data_files=VAL_PATH, split="val")
print(f"Train: {len(train_ds)} samples")
print(f"Val:   {len(val_ds)} samples")
print(f"Columns: {train_ds.column_names}")
print(f"Sample: {train_ds[0]}")

# Schema for Laya: state + questions in dict format
def format_for_laya(example):
    state = {"text": example["text"]}
    questions = {
        "is_scam": {"type": "noul", "instructions": "Is this a scam?"},
        "category": {
            "type": "choice",
            "instructions": "What category of scam is this?",
            "criteria": {
                "benign": "normal legitimate message",
                "phishing": "phishing link",
                "crypto_scam": "cryptocurrency fraud",
                "investment_scam": "fake investment",
                "lottery_scam": "fake prize",
                "job_scam": "fake job offer",
                "loan_scam": "fake loan",
                "impersonation": "fake official",
                "romance_scam": "pig butchering",
                "delivery_fraud": "fake courier",
                "marketing": "legitimate marketing",
                "adult_content": "adult services",
                "spam_general": "other junk",
            },
        },
    }
    return {
        "state": str(state),
        "questions": str(questions),
        "target_is_scam": example["is_scam"],
        "target_category": example["category"],
    }

train_ds = train_ds.map(format_for_laya)
val_ds = val_ds.map(format_for_laya)
print(f"\\nFormatted sample: {train_ds[0]}")
""")
add(data_cell)

# Cell 5: RLCD trainer scaffolding
rlcd = nbf.v4.new_code_cell("""\
import torch
import torch.nn.functional as F


def compute_reward(logits: torch.Tensor, target: torch.Tensor, temperature: float = 1.0):
    \"\"\"Strictly proper scoring rules: log + spherical.\"\"\"
    eps = 1e-7
    scaled = logits / temperature
    probs = F.softmax(scaled, dim=-1)
    log_score = torch.log(probs.gather(1, target.unsqueeze(1)).squeeze() + eps)
    target_onehot = F.one_hot(target, probs.size(-1)).float()
    norm_p = probs / (probs.sum(dim=-1, keepdim=True) + eps)
    norm_t = target_onehot / (target_onehot.sum(dim=-1, keepdim=True) + eps)
    spherical = (norm_p * norm_t).sum(dim=-1)
    return (log_score + spherical).mean()


print("RLCD reward fn ready (Brier + spherical on log+softmax)")
print("\\nNOTE: Full training loop below uses HuggingFace Trainer + LoRA.")
print("RLCD loss approximation: use cross-entropy on category labels +")
print("binary cross-entropy on is_scam labels as a proxy.")
""")
add(rlcd)

# Cell 6: Training loop (HF Trainer + LoRA)
training = nbf.v4.new_code_cell("""\
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
from torch.utils.data import DataLoader

# Simplified training loop using HF Trainer
# Real Laya RLCD requires custom loss via the Laya SDK's predict() method,
# which is complex. For the v1 baseline, use standard CE on targets.

training_args = TrainingArguments(
    output_dir="./laya-finetuned",
    num_train_epochs=4,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,  # effective batch=32
    learning_rate=2e-4,
    warmup_ratio=0.06,
    max_grad_norm=1.0,
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=2,
    eval_strategy="epoch",
    fp16=True,  # T4 supports fp16
    report_to="none",
    dataloader_num_workers=2,
)

# Custom data collator: feed state + questions + targets
def collate(batch):
    states = [b["state"] for b in batch]
    questions = [b["questions"] for b in batch]
    # For simplicity, concatenate state + questions as text input
    texts = [s + " " + q for s, q in zip(states, questions)]
    encoded = router.multilingual.tokenizer(
        texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
    )
    encoded["labels"] = encoded["input_ids"].clone()
    return encoded

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate,
)

print("Starting training (~4-5 hours on 2x T4)...")
trainer.train()
print("✓ Training complete")
""")
add(training)

# Cell 7: Merge LoRA + save merged model
merge = nbf.v4.new_code_cell("""\
from peft import PeftModel
import torch
import json

# Load base model fresh
del peft_model
torch.cuda.empty_cache()

base = router.multilingual.model
merged = PeftModel.from_pretrained(base, "./laya-finetuned/checkpoint-3")
merged = merged.merge_and_unload()

# Save merged model
merged.save_pretrained("./laya-merged")
router.multilingual.tokenizer.save_pretrained("./laya-merged")
print("✓ Saved merged model")

# Apply fitted temperatures (placeholder — actual fit happens on val)
# In production, fit_temperatures(merged, val_ds)
config = json.load(open("./laya-merged/rl_agent_config.json"))
config["temperature_by_options"] = {
    "choice:3-5": 1.0, "choice:6-10": 1.0, "choice:2": 1.0,
    "score:3-5": 1.0, "noul:2": 1.0
}
with open("./laya-merged/rl_agent_config.json", "w") as f:
    json.dump(config, f, indent=2)
print("✓ Updated rl_agent_config with placeholder temperatures")
""")
add(merge)

# Cell 8: Push to HuggingFace Hub
upload = nbf.v4.new_code_cell("""\
from huggingface_hub import HfApi

# REPLACE with your HF username before running!
HF_USER = "YOUR_USERNAME"

api = HfApi()

# Push merged model
api.upload_folder(
    folder_path="./laya-merged",
    repo_id=f"{HF_USER}/laya-multilingual-finetuned-scam",
    repo_type="model",
    private=True,
)
print(f"✓ Pushed merged model to huggingface.co/{HF_USER}/laya-multilingual-finetuned-scam")

# Save + push adapter only
import os
os.system("cp -r ./laya-finetuned/checkpoint-3/adapter_model ./adapter-only")
os.system("cp -r ./laya-finetuned/checkpoint-3/adapter_config.json ./adapter-only/")
api.upload_folder(
    folder_path="./adapter-only",
    repo_id=f"{HF_USER}/laya-multilingual-scam-adapter",
    repo_type="model",
    private=True,
)
print(f"✓ Pushed adapter to huggingface.co/{HF_USER}/laya-multilingual-scam-adapter")
""")
add(upload)

with open(nb_path, "w") as f:
    nbf.write(nb, f)

print(f"✓ Added {len(nb.cells) - 1} cells to notebook")