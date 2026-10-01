"""V2 训练集显式配比（FP 压降核心）。设计见 2026-09-30-laya-fp-reduction-design.md §7.1。

compose_train 取代 v1 的 per-source balance_train：难负/简单负样本优先全保留，
FGRC 提示类压到 10–15%，CCL/Tele/单类别按上限裁剪。
"""
import hashlib
import random
from collections import Counter, defaultdict

TRAIN_TARGET = 45_000
POS_FRACTION = 0.45
CCL_NONREBATE_CAP = 0.20   # CCL 非刷单返利 ≤20% 正样本（rebate 豁免——spec 勘误）
TELE_POS_CAP = 0.10        # TeleAntiFraud ≤10% 正样本
PER_CATEGORY_CAP = 0.30    # 任何单类别 ≤30% 正样本
REBATE_FLOOR = 0.25        # rebate_scam ≥25% 正样本
CHIFRAUD_BENIGN_CAP = 0.10
TIPS_FRACTION = 0.125      # FGRC 提示类目标（区间 0.10–0.15 中点）
GEN_CAP = 0.30             # 生成样本 ≤30% 训练总量

HARDNEG_SOURCES = frozenset({
    "hn_antifraud_propaganda", "hn_financial_notice", "hn_ecommerce_logistics",
    "hn_job_ad", "hn_promotion", "hn_gov_notice", "hn_personal_social",
    "hn_traffic_funnel", "contrastive_pair",
})
GEN_SOURCES = frozenset({
    "hn_financial_notice", "hn_ecommerce_logistics", "hn_job_ad",
    "hn_promotion", "hn_gov_notice", "hn_personal_social",
    "hn_traffic_funnel", "nb_natural_benign", "contrastive_pair",
})
FGRC_TIP_SOURCES = frozenset({"fgrc_scd_sms", "fgrc_scd_dialog"})


def stable_id(text: str, source: str) -> str:
    """与 build_dataset.stable_id 公式完全一致（test_stable_id_matches 强制）。"""
    h = hashlib.sha1(f"{source}::{text}".encode("utf-8")).hexdigest()[:12]
    return f"{source}-{h}"


def _sample(items: list[dict], k: int, rng: random.Random) -> list[dict]:
    if k <= 0:
        return []
    if len(items) <= k:
        return items[:]
    return rng.sample(items, k)


def compose_train(rows: list[dict], eval_ids: set[str],
                  train_target: int = TRAIN_TARGET, seed: int = 42,
                  ) -> tuple[list[dict], dict]:
    """按配比组装训练集。返回 (train_rows, stats)。违反硬约束即抛错。"""
    rng = random.Random(seed)
    pool = [r for r in rows if r.get("id") not in eval_ids]
    pos_all = [r for r in pool if r["is_scam"] == 1]
    neg_all = [r for r in pool if r["is_scam"] == 0]

    n_pos = round(train_target * POS_FRACTION)
    n_neg = train_target - n_pos

    # —— 正样本（预算瀑布）——
    rebate_pool = [r for r in pos_all if r["category"] == "rebate_scam"]
    need_rebate = round(n_pos * REBATE_FLOOR)
    if len(rebate_pool) < need_rebate:
        raise RuntimeError(
            f"rebate_scam 仅 {len(rebate_pool)} 条 < 下限 {need_rebate}"
            f" —— CCL2023 未就绪或被过度去重")
    pos_taken = _sample(rebate_pool, need_rebate, rng)
    taken_ids = {r["id"] for r in pos_taken}
    budget = n_pos - len(pos_taken)

    ccl_rest = [r for r in pos_all if r["source"] == "ccl2023"
                and r["category"] != "rebate_scam" and r["id"] not in taken_ids]
    picked = _sample(ccl_rest, min(round(n_pos * CCL_NONREBATE_CAP), budget), rng)
    pos_taken += picked
    taken_ids |= {r["id"] for r in picked}
    budget -= len(picked)

    tele_rest = [r for r in pos_all if r["source"] == "teleantifraud"
                 and r["id"] not in taken_ids]
    picked = _sample(tele_rest, min(round(n_pos * TELE_POS_CAP), budget), rng)
    pos_taken += picked
    taken_ids |= {r["id"] for r in picked}
    budget -= len(picked)

    cat_cap = round(n_pos * PER_CATEGORY_CAP)
    ccl_nr_cap = round(n_pos * CCL_NONREBATE_CAP)
    tele_cap = round(n_pos * TELE_POS_CAP)
    taken_cat = Counter(r["category"] for r in pos_taken)
    taken = {
        "ccl_nr": sum(1 for r in pos_taken
                      if r["source"] == "ccl2023" and r["category"] != "rebate_scam"),
        "tele": sum(1 for r in pos_taken if r["source"] == "teleantifraud"),
    }

    def _caps_block(r: dict) -> bool:
        """三重跨桶上限：单类别 ≤30% / ccl 非刷单返利 ≤20% / tele ≤10%。"""
        if taken_cat[r["category"]] >= cat_cap:
            return True
        if (r["source"] == "ccl2023" and r["category"] != "rebate_scam"
                and taken["ccl_nr"] >= ccl_nr_cap):
            return True
        if r["source"] == "teleantifraud" and taken["tele"] >= tele_cap:
            return True
        return False

    def _take_eligible(pool: list[dict], k: int) -> None:
        nonlocal budget
        rng.shuffle(pool)
        n = 0
        for r in pool:
            if n >= k or budget <= 0:
                break
            if r["id"] in taken_ids or _caps_block(r):
                continue
            pos_taken.append(r)
            taken_ids.add(r["id"])
            taken_cat[r["category"]] += 1
            if r["source"] == "ccl2023" and r["category"] != "rebate_scam":
                taken["ccl_nr"] += 1
            if r["source"] == "teleantifraud":
                taken["tele"] += 1
            budget -= 1
            n += 1

    # 阶段 A：按可用量比例分配（最大余数法），避免字母序贪心饿死后位类别
    pools: dict[str, list[dict]] = {}
    for r in pos_all:
        if r["id"] in taken_ids or _caps_block(r):
            continue
        pools.setdefault(r["category"], []).append(r)
    eff = {cat: min(len(pool), max(cat_cap - taken_cat.get(cat, 0), 0))
           for cat, pool in pools.items()}
    total_eff = sum(eff.values())
    if budget > 0 and total_eff > 0:
        share_total = min(budget, total_eff)
        raw = {cat: share_total * eff[cat] / total_eff for cat in pools}
        targets = {cat: int(raw[cat]) for cat in pools}
        remainder = share_total - sum(targets.values())
        for cat in sorted(pools, key=lambda c: -(raw[c] - int(raw[c]))):
            if remainder <= 0:
                break
            if targets[cat] < eff[cat]:
                targets[cat] += 1
                remainder -= 1
        for cat in sorted(pools):
            _take_eligible(pools[cat], targets[cat])

    # 阶段 B：按 source 占比回填余量（仍受三重上限）
    if budget > 0:
        leftovers = [r for r in pos_all
                     if r["id"] not in taken_ids and not _caps_block(r)]
        if leftovers:
            src_w = Counter(r["source"] for r in leftovers)
            by_src: dict[str, list[dict]] = {}
            for r in leftovers:
                by_src.setdefault(r["source"], []).append(r)
            for src in sorted(by_src):
                if budget <= 0:
                    break
                share = round(budget * src_w[src] / len(leftovers)) + 1
                _take_eligible(by_src[src], share)

    # 阶段 C：最后兜底（仍受三重上限）
    if budget > 0:
        rest_pos = [r for r in pos_all
                    if r["id"] not in taken_ids and not _caps_block(r)]
        _take_eligible(rest_pos, budget)

    # —— 三重上限跨桶终检（design §7.1：单类别 ≤30% / ccl 非返利 ≤20% / tele ≤10%）——
    for cat, k in taken_cat.items():
        if k > cat_cap + 1:
            raise RuntimeError(
                f"类别 {cat} 占正样本 {k}/{len(pos_taken)} 超过 {PER_CATEGORY_CAP:.0%} 上限")
    if taken["ccl_nr"] > ccl_nr_cap + 1:
        raise RuntimeError(f"ccl 非返利 {taken['ccl_nr']} 超 {CCL_NONREBATE_CAP:.0%} 上限")
    if taken["tele"] > tele_cap + 1:
        raise RuntimeError(f"tele {taken['tele']} 超 {TELE_POS_CAP:.0%} 上限")

    # —— 负样本（贪婪瀑布：难负 > 简单 > 提示类 > ChiFraud > 自然 benign）——
    hard = [r for r in neg_all if r["source"] in HARDNEG_SOURCES]
    simple = [r for r in neg_all if r["source"] == "nb_natural_benign"]
    tips_pool = [r for r in neg_all
                 if r["source"] in FGRC_TIP_SOURCES and r["category"] == "benign"]
    natural = [r for r in neg_all
               if r["source"] not in HARDNEG_SOURCES
               and r["source"] != "nb_natural_benign"
               and r["source"] not in FGRC_TIP_SOURCES]

    neg_taken: list[dict] = []
    nbudget = n_neg
    picked = _sample(hard, min(len(hard), nbudget), rng)
    neg_taken += picked
    nbudget -= len(picked)
    hard_kept = len(picked)
    picked = _sample(simple, min(len(simple), nbudget), rng)
    neg_taken += picked
    nbudget -= len(picked)
    simple_kept = len(picked)
    tips_quota = min(round(n_neg * TIPS_FRACTION), nbudget)
    picked = _sample(tips_pool, tips_quota, rng)
    neg_taken += picked
    nbudget -= len(picked)
    tips_kept = len(picked)
    chifraud = [r for r in natural if r["source"] == "chifraud"]
    natural_other = [r for r in natural if r["source"] != "chifraud"]
    picked = _sample(chifraud, min(round(n_neg * CHIFRAUD_BENIGN_CAP), nbudget), rng)
    neg_taken += picked
    nbudget -= len(picked)
    picked = _sample(natural_other, min(len(natural_other), nbudget), rng)
    neg_taken += picked
    nbudget -= len(picked)

    train = pos_taken + neg_taken
    rng.shuffle(train)

    # —— 生成样本上限 ——
    gen_n = sum(1 for r in train if r["source"] in GEN_SOURCES)
    if gen_n > GEN_CAP * len(train):
        raise RuntimeError(
            f"生成样本 {gen_n}/{len(train)} ({gen_n / len(train):.1%}) "
            f"超过 {GEN_CAP:.0%} 上限 —— 降低生成体裁目标量或提高真实数据配额")

    stats = {
        "n": len(train), "n_pos": len(pos_taken), "n_neg": len(neg_taken),
        "hardneg_kept": hard_kept, "simple_kept": simple_kept,
        "tips_kept": tips_kept,
        "gen_share": gen_n / len(train) if train else 0.0,
        "pos_unfilled": budget, "neg_unfilled": nbudget,
        "pos_by_category": dict(Counter(r["category"] for r in pos_taken)),
        "neg_by_source": dict(Counter(r["source"] for r in neg_taken)),
    }
    return train, stats
