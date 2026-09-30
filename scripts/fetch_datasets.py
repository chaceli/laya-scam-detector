"""Download public scam-detection datasets for fine-tuning.

Uses curl to bypass Hugging Face's SOCKS proxy issues on this machine.

Working datasets (as of 2026-09):
- FGRC-SCD (Abooooo, Chinese telecom fraud, MIT) — FGRC-SCD-sms.zip + FGRC-SCD-dialog.zip
- Him1304/scamshield-scam-detection-data (multilingual EN+job, MIT) — train.csv
- ealvaradob/phishing-dataset (English URL/SMS/email, research) — combined_reduced.json
- fl-wxiao/FBS_SMS_Dataset (Chinese fake base station, research) — git clone
- ucirvine/sms_spam (English, public domain) — already in datasets/public.jsonl

Skipped due to 401 (gated, requires HF auth):
- vichetkao/Scam_Message_9_Language
- M-Arjun/SpamShield-Datasets
"""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

RAW_DIR = Path("datasets/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


DATASETS = [
    {
        "name": "fgrc_scd_sms",
        "url": "https://huggingface.co/datasets/Abooooo/FGRC-SCD/resolve/main/FGRC-SCD-sms.zip",
        "format": "zip",
    },
    {
        "name": "fgrc_scd_dialog",
        "url": "https://huggingface.co/datasets/Abooooo/FGRC-SCD/resolve/main/FGRC-SCD-dialog.zip",
        "format": "zip",
    },
    {
        "name": "scamshield",
        "url": "https://huggingface.co/datasets/Him1304/scamshield-scam-detection-data/resolve/main/train.csv",
        "format": "csv",
    },
    {
        "name": "ealvaradob_phishing",
        "url": "https://huggingface.co/datasets/ealvaradob/phishing-dataset/resolve/main/combined_reduced.json",
        "format": "json",
    },
]


FBS_SMS_GIT_URL = "https://github.com/fl-wxiao/FBS_SMS_Dataset.git"
FBS_SMS_TARGET = RAW_DIR / "fbs_sms"

CCL2023_GIT_URL = "https://github.com/GJSeason/CCL2023-FCC.git"
CHIFRAUD_GIT_URL = "https://github.com/xuemingxxx/ChiFraud.git"
TELE_ANTIFRAUD_TREE_API = "https://huggingface.co/api/datasets/JimmyMa99/TeleAntiFraud/tree/main"
TELE_ANTIFRAUD_BASE = "https://huggingface.co/datasets/JimmyMa99/TeleAntiFraud/resolve/main"

TELE_FILES: list[str] = []  # Task 4 探测后回填，例如 ["data/train.jsonl"]

PHISHING_EMAIL_DATASET = "naserabdullahalam/phishing-email-dataset"


def curl_download(url: str, target: Path) -> None:
    if target.exists() and target.stat().st_size > 1000:
        print(f"  ✓ {target.name} already exists ({target.stat().st_size:,} bytes)")
        return
    print(f"Downloading {url[:80]}... → {target.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["curl", "-L", "-m", "1200", "-o", str(target), url],
        check=False,
    )
    if result.returncode != 0:
        print(f"  ✗ curl failed for {url}", file=sys.stderr)
        target.unlink(missing_ok=True)


def git_clone(git_url: str, target: Path) -> None:
    if target.exists() and any(target.iterdir()):
        print(f"  ✓ {target.name} already cloned")
        return
    print(f"Cloning {git_url} → {target.name}")
    subprocess.run(
        ["git", "clone", "--depth", "1", git_url, str(target)],
        check=False,
    )


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract a zip into a target directory."""
    if not zip_path.exists():
        return
    print(f"  Extracting {zip_path.name} → {extract_to}")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_to)


def main() -> int:
    for ds in DATASETS:
        print(f"\n[{ds['name']}]")
        target = RAW_DIR / f"{ds['name']}.{ds['format']}"
        curl_download(ds["url"], target)

    print(f"\n[{FBS_SMS_TARGET.name}]")
    git_clone(FBS_SMS_GIT_URL, FBS_SMS_TARGET)

    print("\n[ccl2023]")
    git_clone(CCL2023_GIT_URL, RAW_DIR / "ccl2023")

    print("\n[chifraud]")
    git_clone(CHIFRAUD_GIT_URL, RAW_DIR / "chifraud")

    print("\n[teleantifraud]")
    if TELE_FILES:
        for relpath in TELE_FILES:
            curl_download(f"{TELE_ANTIFRAUD_BASE}/{relpath}",
                          RAW_DIR / "teleantifraud" / Path(relpath).name)
    else:
        print("  ! TELE_FILES 未配置。运行: curl -s " + TELE_ANTIFRAUD_TREE_API)
        print("    从返回 JSON 中挑数据文件路径填入 TELE_FILES 后重跑。")

    print("\n[phishing_email]")
    try:
        import kagglehub
        cache = kagglehub.dataset_download(PHISHING_EMAIL_DATASET)
        target = RAW_DIR / "phishing_email"
        target.mkdir(parents=True, exist_ok=True)
        for f in Path(cache).glob("*"):
            if f.is_file():
                shutil.copy(f, target / f.name)
        print("  ✓ kagglehub -> datasets/raw/phishing_email")
    except Exception as e:
        print(f"  ✗ kagglehub 跳过（凭据/网络）: {e}")

    # Extract FGRC-SCD zips
    for z in ["fgrc_scd_sms", "fgrc_scd_dialog"]:
        zip_path = RAW_DIR / f"{z}.zip"
        extract_to = RAW_DIR / z
        extract_zip(zip_path, extract_to)

    print("\n✓ All datasets downloaded")
    return 0


if __name__ == "__main__":
    sys.exit(main())