"""Tests for category mapping module."""
import pytest

from schemas.scam_categories import (
    CANONICAL_CATEGORIES,
    CATEGORY_MAP,
    normalize_label,
    is_valid_category,
    label_to_index,
    index_to_label,
)


class TestCanonicalCategories:
    def test_exactly_13_categories(self):
        assert len(CANONICAL_CATEGORIES) == 13

    def test_benign_is_first(self):
        assert CANONICAL_CATEGORIES[0] == "benign"

    def test_all_unique(self):
        assert len(set(CANONICAL_CATEGORIES)) == 13


class TestNormalizeLabel:
    def test_known_spamshield_labels(self):
        assert normalize_label("spam") == "spam_general"
        assert normalize_label("crypto") == "crypto_scam"
        assert normalize_label("job_scam") == "job_scam"
        assert normalize_label("ham") == "benign"

    def test_known_fbs_sms_labels(self):
        assert normalize_label("AD:Loan") == "loan_scam"
        assert normalize_label("FR:Phishing(Bank)") == "phishing"
        assert normalize_label("IL:Gambling") == "lottery_scam"
        assert normalize_label("IL:Fake_ID_and_invoice") == "impersonation"

    def test_known_fgrc_scd_labels(self):
        assert normalize_label("low_risk_sms") == "benign"
        assert normalize_label("phishing_link") == "phishing"

    def test_known_ealvaradob_labels(self):
        assert normalize_label("phishing") == "phishing"
        assert normalize_label("legitimate") == "benign"

    def test_unknown_label_falls_back_to_spam_general(self):
        assert normalize_label("totally_new_label_xyz") == "spam_general"

    def test_idempotent_for_canonical(self):
        for c in CANONICAL_CATEGORIES:
            assert normalize_label(c) == c


class TestIndexConversion:
    def test_round_trip(self):
        for label in CANONICAL_CATEGORIES:
            idx = label_to_index(label)
            assert index_to_label(idx) == label

    def test_invalid_label_raises(self):
        with pytest.raises(ValueError):
            label_to_index("nonexistent")

    def test_invalid_index_raises(self):
        with pytest.raises(ValueError):
            index_to_label(99)


class TestIsValidCategory:
    def test_canonical_is_valid(self):
        for c in CANONICAL_CATEGORIES:
            assert is_valid_category(c) is True

    def test_non_canonical_invalid(self):
        assert is_valid_category("spam") is False
        assert is_valid_category("spam_general") is True