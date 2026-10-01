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
    "rebate_scam",       # v2: 刷单返利（公安口径发案量第一）
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


# CCL2023-FCC 严格映射（loader 逐条断言，未覆盖即报错，不走 spam_general 兜底）
CCL2023_LABEL_MAP: dict[str, str] = {
    "刷单返利类": "rebate_scam",
    "冒充电商物流客服类": "delivery_fraud",
    "虚假网络投资理财类": "investment_scam",
    "贷款、代办信用卡类": "loan_scam",
    "虚假征信类": "loan_scam",
    "虚假购物、服务类": "phishing",
    "冒充公检法及政府机关类": "impersonation",
    "冒充领导、熟人类": "impersonation",
    "网络游戏产品虚假交易类": "spam_general",
    "网络婚恋、交友类（非虚假网络投资理财类）": "romance_scam",
    "冒充军警购物类诈骗": "impersonation",
    "网黑案件": "spam_general",
}

# 单一来源注入：normalize_label 经 CATEGORY_MAP 消费 CCL 12 类（用户裁定 2026-09-30）
CATEGORY_MAP.update(CCL2023_LABEL_MAP)

# ChiFraud（灰产供给侧视角）：仅地下贷款可映射，其余在 loader 中显式丢弃
CHIFRAUD_SCAM_MAP: dict[str, str] = {
    "地下贷款": "loan_scam",
    "地下贷款类": "loan_scam",
}
CHIFRAUD_BENIGN_LABELS: frozenset[str] = frozenset(
    {"正常", "benign", "normal", "合法", "非诈骗"})

# TeleAntiFraud-28k：正常通话标签候选（Task 4 探测后按实际修正）
TELE_NORMAL_LABELS: frozenset[str] = frozenset(
    {"normal", "benign", "正常", "非诈骗", "ham"})