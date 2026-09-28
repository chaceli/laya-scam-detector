"""Category mapping from source dataset labels to our unified 13-class schema.

Each public dataset uses different label vocabulary. This module provides:
1. The 13-class canonical taxonomy (CANONICAL_CATEGORIES)
2. Maps source labels → canonical labels (CATEGORY_MAP)
3. Functions to normalize a free-form label to the canonical 13

Sources we normalize:
- SpamShield-Datasets (M-Arjun): spam/phishing/crypto/marketing/job_scam/giveaway/adult/promo/normal
- FBS_SMS_Dataset (fl-wxiao): AD:Loan/AD:Network_service/AD:Other/FR:Financial/FR:Phishing(Bank)/
                              FR:Phishing(Other)/FR:Other/IL:Escort_service/IL:Fake_ID_and_invoice/
                              IL:Gambling/IL:Political_propaganda/Other
- FGRC-SCD: low_risk_sms/high_risk_sms/fraud_call/phishing_link (custom)
- ealvaradob: phishing/url_phishing/html_phishing/legitimate
- UC Irvine SMS Spam: ham/spam
- vichetkao/Scam_Message_9_Language: ham/spam (binary only)
- Handwritten (existing): benign / delivery_fraud / phishing / romance_scam / investment_scam /
                            impersonation / lottery_scam / loan_scam
"""
from __future__ import annotations


CANONICAL_CATEGORIES = [
    "benign",
    "phishing",
    "crypto_scam",
    "investment_scam",
    "lottery_scam",
    "job_scam",
    "loan_scam",
    "impersonation",
    "romance_scam",
    "delivery_fraud",
    "marketing",
    "adult_content",
    "spam_general",
]

CANONICAL_TO_INDEX = {c: i for i, c in enumerate(CANONICAL_CATEGORIES)}


CATEGORY_MAP: dict[str, str] = {
    # SpamShield (English + others)
    "spam": "spam_general",
    "phishing": "phishing",
    "crypto": "crypto_scam",
    "marketing": "marketing",
    "job_scam": "job_scam",
    "giveaway": "lottery_scam",
    "adult": "adult_content",
    "promo": "marketing",
    "normal": "benign",
    "ham": "benign",
    "legitimate": "benign",

    # FBS_SMS_Dataset (Chinese)
    "AD:Loan": "loan_scam",
    "AD:Network_service": "spam_general",
    "AD:Other": "spam_general",
    "FR:Financial": "investment_scam",
    "FR:Phishing(Bank)": "phishing",
    "FR:Phishing(Other)": "phishing",
    "FR:Other": "spam_general",
    "IL:Escort_service": "adult_content",
    "IL:Fake_ID_and_invoice": "impersonation",
    "IL:Gambling": "lottery_scam",
    "IL:Political_propaganda": "spam_general",
    "Other": "spam_general",

    # FGRC-SCD (Chinese telecom fraud)
    "low_risk_sms": "benign",
    "high_risk_sms": "spam_general",
    "fraud_call": "impersonation",
    "phishing_link": "phishing",

    # ealvaradob phishing dataset
    "url_phishing": "phishing",
    "html_phishing": "phishing",

    # Handwritten (existing project dataset)
    "benign": "benign",
    "delivery_fraud": "delivery_fraud",
    "romance_scam": "romance_scam",
    "investment_scam": "investment_scam",
    "impersonation": "impersonation",
    "lottery_scam": "lottery_scam",
    "loan_scam": "loan_scam",
}


def normalize_label(raw_label: str) -> str:
    """Map any source label to canonical 13-class category.

    Falls back to 'spam_general' for unknown source labels (better than
    dropping the sample, which would lose training data). Canonical
    labels pass through unchanged.
    """
    if raw_label in CANONICAL_TO_INDEX:
        return raw_label
    return CATEGORY_MAP.get(raw_label, "spam_general")


def is_valid_category(label: str) -> bool:
    return label in CANONICAL_TO_INDEX


def label_to_index(label: str) -> int:
    if label not in CANONICAL_TO_INDEX:
        raise ValueError(f"unknown category: {label!r}")
    return CANONICAL_TO_INDEX[label]


def index_to_label(idx: int) -> str:
    if not 0 <= idx < len(CANONICAL_CATEGORIES):
        raise ValueError(f"index out of range: {idx}")
    return CANONICAL_CATEGORIES[idx]