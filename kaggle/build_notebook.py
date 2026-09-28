"""Generate the Kaggle notebook for Laya multilingual fine-tuning."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell(
        "# Laya Multilingual Fine-tuning for Scam Detection\n"
        "\n"
        "Trains LoRA adapter on `convaiinnovations/laya-multilingual` with "
        "multi-source scam data. Targets:\n"
        "\n"
        "- Chinese is_scam accuracy >= 0.90 (baseline 0.840)\n"
        "- Chinese recall >= 0.95\n"
        "- 13-class accuracy >= 0.70\n"
        "- Calibration ECE < 0.05\n"
        "\n"
        "**Hardware**: 2x T4 GPU (free Kaggle tier)\n"
        "**Time**: ~4-5 hours\n"
        "**Output**: merged model + LoRA adapter pushed to private HF Hub"
    ),
]

with open("kaggle/laya_finetune_multilingual.ipynb", "w") as f:
    nbf.write(nb, f)

print("✓ Notebook scaffold created")