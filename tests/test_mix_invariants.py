"""compose_train 不变量的对抗性测试：用恶意输入实际攻击每条约束。

纸面评审容易漏；这里构造能让实现"想当然"失败的输入形态，验证约束真的守得住。
"""
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from dataset_mix import (  # noqa: E402
    CCL_NONREBATE_CAP, GEN_CAP, PER_CATEGORY_CAP, POS_FRACTION,
    REBATE_FLOOR, TELE_POS_CAP, compose_train, stable_id,
)


def mk(text, is_scam, source, category=None):
    return {
        "id": f"{source}-{abs(hash(text)) % 10**8}",
        "text": text, "is_scam": is_scam, "risk": 4 if is_scam else 1,
        "category": category or ("benign" if not is_scam else "spam_general"),
        "language": "zh", "source": source,
    }


def parse_rows(rows, eval_ids, target=4000, seed=42):
    return compose_train(rows, eval_ids=eval_ids, train_target=target, seed=seed)


def assert_caps_hold(train, stats, target):
    """上限相对**目标** n_pos 判定，不是实际产出。

    实际 n_pos 可能因供给不足而欠填（pos_unfilled > 0）；此时用实际值算上限会
    把合法结果误判为越限（例如唯一可用类别恰好填到上限时）。
    """
    pos = [r for r in train if r["is_scam"] == 1]
    assert len(pos) == stats["n_pos"]
    n_pos_target = round(target * POS_FRACTION)
    assert stats["n_pos"] <= n_pos_target, "正样本数超过目标"
    cats = Counter(r["category"] for r in pos)
    cap = round(n_pos_target * PER_CATEGORY_CAP)
    for c, k in cats.items():
        assert k <= cap + 1, f"类别 {c} = {k} 超过 cap {cap}"
    ccl_nr = sum(1 for r in pos
                 if r["source"] == "ccl2023" and r["category"] != "rebate_scam")
    assert ccl_nr <= round(n_pos_target * CCL_NONREBATE_CAP) + 1, f"ccl_nr={ccl_nr} 越限"
    tele = sum(1 for r in pos if r["source"] == "teleantifraud")
    assert tele <= round(n_pos_target * TELE_POS_CAP) + 1, f"tele={tele} 越限"
    gen = sum(1 for r in train
              if r["source"].startswith(("hn_", "nb_", "contrastive")))
    assert gen <= GEN_CAP * len(train) + 1e-9, f"gen={gen}/{len(train)} 越限"


def _negatives(n, source="ealvaradob"):
    return [mk(f"普通短信{i}：会议通知。", 0, source) for i in range(n)]


def _rebates(n):
    return [mk(f"刷单返利案{i}：受害人陈述。", 1, "ccl2023", "rebate_scam")
            for i in range(n)]


class TestCapsUnderAttack:
    def test_single_category_domination_capped(self):
        """单一类别占满全部正样本 —— 只能填到 30% 上限，不允许填满 45% 目标。"""
        rows = _rebates(3000) + _negatives(3000)
        train, stats = parse_rows(rows, set(), target=4000)
        assert_caps_hold(train, stats, 4000)
        cap = round(round(4000 * POS_FRACTION) * PER_CATEGORY_CAP)
        assert stats["n_pos"] == cap
        assert stats["pos_unfilled"] > 0, "供给受限时应记录欠填量"

    def test_ccl_nonrebate_spread_across_many_categories_capped(self):
        """ccl 非返利行伪装成很多不同类别（每类都不触发单类 cap）—— 合计仍须 ≤20%。"""
        rows = _rebates(2500)
        cats = ["phishing", "loan_scam", "impersonation", "romance_scam",
                "job_scam", "delivery_fraud", "spam_general", "adult_content",
                "crypto_scam", "marketing", "investment_scam", "lottery_scam"]
        for c in cats:
            rows += [mk(f"ccl {c} {i}：受害人陈述。", 1, "ccl2023", c)
                     for i in range(400)]
        rows += _negatives(4000)
        train, stats = parse_rows(rows, set(), target=4000)
        assert_caps_hold(train, stats, 4000)

    def test_tele_rows_hiding_in_a_rare_category_capped(self):
        """tele 行藏在稀有类别里（类别余量很大）—— tele 源总量仍须 ≤10%。"""
        rows = _rebates(2500)
        for c in ("crypto_scam", "marketing", "adult_content", "job_scam"):
            rows += [mk(f"tele {c} {i}：来电内容。", 1, "teleantifraud", c)
                     for i in range(500)]
        rows += _negatives(4000)
        train, stats = parse_rows(rows, set(), target=4000)
        assert_caps_hold(train, stats, 4000)

    def test_rebate_pool_below_floor_raises(self):
        """rebate 供给不足下限必须抛错（这是 Task 13 的门禁语义）。"""
        rows = [mk(f"诈骗{i}", 1, "fgrc_scd_sms", "phishing") for i in range(500)]
        rows += _negatives(3000)
        with pytest.raises(RuntimeError, match="rebate_scam"):
            parse_rows(rows, set(), target=4000)

    def test_all_generated_negatives_raise_gen_cap(self):
        """负样本几乎全是生成样本 —— 必须抛 gen cap 错，而不是悄悄训出偏斜模型。"""
        rows = _rebates(2500)
        for c in ("crypto_scam", "marketing"):
            rows += [mk(f"x {c} {i}", 1, "fgrc_scd_sms", c) for i in range(500)]
        rows += [mk(f"生成难负{i}：真实业务通知。", 0, "hn_financial_notice")
                 for i in range(3000)]
        with pytest.raises(RuntimeError, match="生成样本"):
            parse_rows(rows, set(), target=4000)


class TestEvalExclusion:
    def test_eval_ids_never_appear_in_output(self):
        """评测集 id 必须从输出中彻底消失（含正负两侧）。"""
        rows = _rebates(2000)
        rows += [mk(f"诈骗{i}", 1, "fgrc_scd_sms", "phishing") for i in range(600)]
        rows += [mk(f"难负{i}", 0, "hn_financial_notice") for i in range(600)]
        rows += _negatives(2000)
        poison = {r["id"] for r in rows[::3]}   # 每三条投毒一条
        train, _ = parse_rows(rows, poison, target=4000)
        out_ids = {r["id"] for r in train}
        assert not (out_ids & poison), f"泄漏 {len(out_ids & poison)} 条评测样本"

    def test_determinism_same_seed_same_output(self):
        """同 seed 必须完全复现（已提交的 eval 切分依赖这个性质）。"""
        rows = _rebates(2000) + _negatives(2000)
        a, _ = parse_rows(rows, set(), target=3000, seed=7)
        b, _ = parse_rows(rows, set(), target=3000, seed=7)
        assert [r["id"] for r in a] == [r["id"] for r in b]
        c, _ = parse_rows(rows, set(), target=3000, seed=8)
        assert [r["id"] for r in a] != [r["id"] for r in c]

    def test_duplicate_ids_deduplicated(self):
        """重复 id（同一行出现多次）不得让计数虚高或重复入池。"""
        base = _rebates(2000) + _negatives(2000)
        rows = base + base[:500]                # 注入 500 条完全重复
        train, _ = parse_rows(rows, set(), target=3000, seed=3)
        ids = [r["id"] for r in train]
        assert len(ids) == len(set(ids)), f"输出含 {len(ids)-len(set(ids))} 条重复"


class TestEdgeShapes:
    def test_tiny_target_still_respects_caps(self):
        """极小 train_target（n_pos 个位数）时不应崩溃或越限。"""
        rows = _rebates(200) + _negatives(200)
        train, stats = parse_rows(rows, set(), target=20, seed=1)
        assert_caps_hold(train, stats, 20)

    def test_negatives_scarce_underfills_without_corrupting_caps(self):
        """负样本远不足时允许欠填，但已有部分仍须守上限。"""
        rows = _rebates(3000) + _negatives(120)
        train, stats = parse_rows(rows, set(), target=4000, seed=5)
        assert_caps_hold(train, stats, 4000)
        assert stats["neg_unfilled"] > 0

    def test_no_negative_rows_at_all(self):
        """完全没有负样本：应只产出正样本且 caps 成立。"""
        rows = _rebates(3000)
        train, stats = parse_rows(rows, set(), target=4000, seed=9)
        assert all(r["is_scam"] == 1 for r in train)
        assert_caps_hold(train, stats, 4000)

    def test_rebate_exactly_at_floor_boundary(self):
        """rebate 供给恰好等于下限时必须通过（边界不能被误判为不足）。"""
        target = 4000
        need = round(round(target * POS_FRACTION) * REBATE_FLOOR)
        rows = _rebates(need) + _negatives(2000)
        train, stats = parse_rows(rows, set(), target=target, seed=2)
        assert_caps_hold(train, stats, 4000)
        rebate = sum(1 for r in train if r["category"] == "rebate_scam")
        assert rebate >= need - 1


class TestEvalSetExclusion:
    """评测集必须按归一化文本排除，不能只靠 id。

    stable_id 把 source 一起哈希，评测行与训练行 source 不同（rebate_eval_real
    vs syn_rebate），因此 id 匹配永远不触发 —— 只传 eval_ids 会让评测文本静默
    混入训练，使该类召回变成自测。
    """

    LEAK = "我被人拉去做刷单返利任务，垫付了三千多"

    def test_id_alone_cannot_exclude_different_source(self):
        """锁定前提：跨 source 的同一句话，id 必然不同。"""
        from dataset_mix import stable_id
        assert stable_id(self.LEAK, "rebate_eval_real") != \
            stable_id(self.LEAK, "syn_rebate")

    def test_eval_text_excluded_despite_differing_id(self):
        rows = _rebates(3000) + _negatives(2000) + [
            mk(self.LEAK, 1, "syn_rebate", "rebate_scam")
        ]
        train, _ = compose_train(
            rows, eval_ids={stable_id(self.LEAK, "rebate_eval_real")},
            eval_texts={self.LEAK}, train_target=4000, seed=11,
        )
        assert self.LEAK not in {r["text"] for r in train}

    def test_punctuation_and_space_variants_excluded(self):
        """全角/空格/零宽差异不得绕过排除。"""
        variants = [
            "我被人拉去做刷单返利任务，垫付了三千多",
            "我被人拉去做刷单返利任务,垫付了三千多",
            "我被人拉去做刷单返利任务 垫付了三千多",
            "我被人拉去做刷单返利任务，垫付了三千多 ",
        ]
        rows = _rebates(3000) + _negatives(2000) + [
            mk(v, 1, "syn_rebate", "rebate_scam") for v in variants
        ]
        train, _ = compose_train(
            rows, eval_ids=set(), eval_texts={variants[0]},
            train_target=4000, seed=13,
        )
        from dataset_mix import _norm_text
        kept = {_norm_text(r["text"]) for r in train}
        assert _norm_text(variants[0]) not in kept

    def test_no_eval_texts_keeps_backward_compat(self):
        """eval_texts 缺省时行为不变（老调用方不会被打断）。"""
        rows = _rebates(3000) + _negatives(2000)
        train, stats = compose_train(rows, eval_ids=set(),
                                     train_target=4000, seed=17)
        assert_caps_hold(train, stats, 4000)
