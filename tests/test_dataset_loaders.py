"""Loader tests against small fixtures (no network, no real data)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from dataset_loaders import (  # noqa: E402
    despace_cjk, load_ccl2023, load_chifraud,
    load_teleantifraud, load_phishing_email,
)
from schemas.scam_categories import CCL2023_LABEL_MAP  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures"


class TestDespaceCjk:
    def test_removes_space_between_cjk(self):
        assert despace_cjk("今 天 有 雨") == "今天有雨"

    def test_keeps_cjk_latin_boundary(self):
        assert despace_cjk("验证码 K888，勿泄露") == "验证码 K888，勿泄露"


class TestCclLoader:
    def test_fixture_rows(self, monkeypatch):
        monkeypatch.setattr("dataset_loaders.RAW", FIX)
        rows = load_ccl2023()
        assert len(rows) == 2
        assert any(r["category"] == "rebate_scam" for r in rows)
        assert all(r["is_scam"] == 1 and r["risk"] == 4
                   and r["source"] == "ccl2023" and r["language"] == "zh" for r in rows)

    def test_returns_empty_when_dir_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr("dataset_loaders.RAW", tmp_path)
        assert load_ccl2023() == []


class TestChifraudLoader:
    def test_whitelist_only(self, monkeypatch):
        monkeypatch.setattr("dataset_loaders.RAW", FIX)
        rows = load_chifraud()
        assert any(r["is_scam"] == 1 and r["category"] == "loan_scam" for r in rows)
        assert any(r["is_scam"] == 0 and r["category"] == "benign" for r in rows)
        assert not any("信用卡代还" in r["text"] for r in rows)  # 违规提现显式丢弃


class TestTeleLoader:
    def test_binary_map(self, monkeypatch):
        monkeypatch.setattr("dataset_loaders.RAW", FIX)
        rows = load_teleantifraud()
        assert any(r["is_scam"] == 1 and r["category"] == "spam_general" for r in rows)
        assert any(r["is_scam"] == 0 and r["category"] == "benign" for r in rows)

    def test_returns_empty_when_dir_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr("dataset_loaders.RAW", tmp_path)
        assert load_teleantifraud() == []


class TestPhishingLoader:
    def test_only_safe_rows(self, monkeypatch):
        monkeypatch.setattr("dataset_loaders.RAW", FIX)
        rows = load_phishing_email()
        assert len(rows) == 1
        assert rows[0]["is_scam"] == 0 and rows[0]["language"] == "en"

    def test_combined_only_when_present(self, monkeypatch, tmp_path):
        base = tmp_path / "phishing_email"
        base.mkdir()
        (base / "phishing_email.csv").write_text("text_combined,label\ncombined safe,0\n")
        (base / "CEAS_08.csv").write_text("text_combined,label\nper-source safe,0\n")
        monkeypatch.setattr("dataset_loaders.RAW", tmp_path)
        rows = load_phishing_email()
        assert len(rows) == 1
        assert rows[0]["text"] == "combined safe"

    def test_falls_back_to_per_source_when_combined_missing(self, monkeypatch, tmp_path):
        base = tmp_path / "phishing_email"
        base.mkdir()
        (base / "CEAS_08.csv").write_text("text_combined,label\nper-source safe,0\n")
        (base / "Enron.csv").write_text("text_combined,label\nsecond safe,0\n")
        monkeypatch.setattr("dataset_loaders.RAW", tmp_path)
        rows = load_phishing_email()
        assert len(rows) == 2
        texts = {r["text"] for r in rows}
        assert texts == {"per-source safe", "second safe"}


class TestMapCompleteness:
    def test_fixture_labels_in_strict_map(self):
        import json
        for rec in json.loads((FIX / "ccl2023" / "train_sample.json").read_text()):
            assert rec["案件类别"] in CCL2023_LABEL_MAP
