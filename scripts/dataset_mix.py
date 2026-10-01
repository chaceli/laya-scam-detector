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

    others = [r for r in pos_all if r["id"] not in taken_ids]
    by_cat = defaultdict(list)
    for r in others:
        by_cat[r["category"]].append(r)
    cat_cap = round(n_pos * PER_CATEGORY_CAP)
    ccl_nr_cap = round(n_pos * CCL_NONREBATE_CAP)
    taken_cat = Counter(r["category"] for r in pos_taken)  # 跨桶：rebate/ccl-nr/tele 桶已占额度
    taken_ccl_nr = sum(1 for r in pos_taken
                       if r["source"] == "ccl2023" and r["category"] != "rebate_scam")
    for cat in sorted(by_cat):
        if budget <= 0:
            break
        remaining_cat = max(cat_cap - taken_cat.get(cat, 0), 0)
        if remaining_cat == 0:
            continue
        # 跨桶 ccl_nr cap：cap 已满时排除 ccl 非刷单返利行（防伪重复文体跨桶越界）
        pool = by_cat[cat]
        if taken_ccl_nr >= ccl_nr_cap:
            pool = [r for r in pool
                    if not (r["source"] == "ccl2023"
                            and r["category"] != "rebate_scam")]
        if not pool:
            continue
        picked = _sample(pool, min(remaining_cat, budget), rng)
        pos_taken += picked
        taken_ids |= {r["id"] for r in picked}
        taken_cat[cat] += len(picked)
        taken_ccl_nr += sum(1 for r in picked
                            if r["source"] == "ccl2023"
                            and r["category"] != "rebate_scam")
        budget -= len(picked)

    if budget > 0:  # 比例填充剩余（按 source 占比；单类别余量约束）
        taken_cat = Counter(r["category"] for r in pos_taken)
        leftovers = [r for r in pos_all if r["id"] not in taken_ids
                     and taken_cat[r["category"]] < cat_cap]
        if leftovers:
            src_w = Counter(r["source"] for r in leftovers)
            by_src = defaultdict(list)
            for r in leftovers:
                by_src[r["source"]].append(r)
            for src in sorted(by_src):
                if budget <= 0:
                    break
                share = min(round(budget * src_w[src] / len(leftovers)) + 1, budget)
                rng.shuffle(by_src[src])
                picked = []
                for r in by_src[src]:
                    if len(picked) >= share or budget <= 0:
                        break
                    if taken_cat[r["category"]] < cat_cap:
                        picked.append(r)
                        taken_cat[r["category"]] += 1
                pos_taken += picked
                taken_ids |= {r["id"] for r in picked}
                budget -= len(picked)
    if budget > 0:  # 最后兜底（仍受单类别余量约束）
        taken_cat = Counter(r["category"] for r in pos_taken)
        rest_pos = [r for r in pos_all if r["id"] not in taken_ids
                    and taken_cat[r["category"]] < cat_cap]
        picked = _sample(rest_pos, min(len(rest_pos), budget), rng)
        pos_taken += picked
        budget -= len(picked)

    # —— 单类别 cap 跨桶终检（design §7.1：任何单类别 ≤30% 正样本）——
    for cat, k in Counter(r["category"] for r in pos_taken).items():
        if k > cat_cap + 1:
            raise RuntimeError(
                f"类别 {cat} 占正样本 {k}/{len(pos_taken)} "
                f"({k / len(pos_taken):.1%}) 超过 {PER_CATEGORY_CAP:.0%} 上限"
                f" —— 调整配额或数据源配比")

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
