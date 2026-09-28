"""Verify mmBERT-base tokenization efficiency on Chinese text.

The English ModernBERT-large backbone shreds CJK into 1-character-per-token
which hurts inference speed and accuracy. We verify that mmBERT-base
(laya-multilingual) achieves better Chinese tokenization.

Acceptance: chars/token >= 1.5 on a sample of Chinese text.
If true (likely), we can proceed with the multilingual base.
If false (< 1.5), we should consider alternative backbones.
"""
import os

import pytest

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


CHINESE_SAMPLES = [
    "您好，我是XX快递客服，您有一个包裹丢失需要理赔。",
    "【最高人民检察院】您涉嫌洗钱，请将资金转入安全账户。",
    "恭喜您被抽中二等奖，奖金10万元，请尽快联系客服领取。",
    "宝贝儿，我在这边投资了一个平台，稳赚不赔，先投5000试试。",
    "您的银行账户将被停用，请点击链接重新认证身份。",
]


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not available")
@pytest.mark.skipif(
    not os.environ.get("TEST_MULTILINGUAL_TOKENIZER"),
    reason="Gated behind TEST_MULTILINGUAL_TOKENIZER env var — requires HF model download",
)
class TestTokenizationQuality:
    def test_mmbert_chinese_efficiency(self):
        """mmBERT-base should achieve >= 1.5 chars/token on Chinese."""
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("answerdotai/mmbert-base")
        total_chars = sum(len(s) for s in CHINESE_SAMPLES)
        total_tokens = sum(len(tok.encode(s)) for s in CHINESE_SAMPLES)
        ratio = total_chars / total_tokens
        assert ratio >= 1.5, (
            f"mmBERT-base Chinese tokenization ratio {ratio:.2f} "
            f"(< 1.5 chars/token indicates inefficient CJK handling)"
        )

    def test_modernbert_chinese_efficiency_baseline(self):
        """ModernBERT-large (English) baseline for comparison."""
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
        total_chars = sum(len(s) for s in CHINESE_SAMPLES)
        total_tokens = sum(len(tok.encode(s)) for s in CHINESE_SAMPLES)
        ratio = total_chars / total_tokens
        # ModernBERT typically gets 0.5-0.8 chars/token on Chinese
        # We expect mmBERT > ModernBERT
        print(f"\nModernBERT-large Chinese ratio: {ratio:.2f} chars/token")
        # Just informational — no assertion