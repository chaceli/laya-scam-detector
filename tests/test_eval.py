"""Tests for eval module: state normalization and thresholded prediction."""
import pytest

from src.eval import _state_from_record, _thresholded_is_scam


class TestStateFromRecord:
    def test_single_message_returns_state(self):
        rec = {"type": "single", "state": "Hello world"}
        assert _state_from_record(rec) == "Hello world"

    def test_multi_turn_formats_with_speaker_tags(self):
        rec = {
            "type": "multi_turn",
            "turns": [
                {"role": "scammer", "text": "Hi"},
                {"role": "victim", "text": "Hello"},
            ],
        }
        out = _state_from_record(rec)
        assert "[S1] Hi" in out
        assert "[V2] Hello" in out

    def test_missing_role_uses_question_mark(self):
        rec = {
            "type": "multi_turn",
            "turns": [{"role": "", "text": "Hello"}],
        }
        out = _state_from_record(rec)
        assert "[?1] Hello" in out

    def test_single_with_no_state_returns_empty(self):
        rec = {"type": "single"}
        assert _state_from_record(rec) == ""


class TestThresholdedIsScam:
    def test_above_threshold_returns_1(self):
        assert _thresholded_is_scam(0.6) == 1

    def test_below_threshold_returns_0(self):
        assert _thresholded_is_scam(0.4) == 0

    def test_at_threshold_returns_1(self):
        assert _thresholded_is_scam(0.5) == 1

    def test_zero_returns_0(self):
        assert _thresholded_is_scam(0.0) == 0

    def test_one_returns_1(self):
        assert _thresholded_is_scam(1.0) == 1