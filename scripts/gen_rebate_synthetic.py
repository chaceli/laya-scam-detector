"""Combinatorial generator for rebate_scam (刷单返利) positive training rows.

The CCL2023 competition dataset (the original rebate_scam source) is not
obtainable, so this script synthesises the class from hand-authored scam
pattern templates. Row format matches build_dataset.load_* output:
  text / is_scam / risk / category / language / source / generator
`id` is filled later by build_dataset.stable_id(text, source).

The real eval set must NOT come from this generator (see
datasets/rebate_eval_real.jsonl) so that the model's rebate recall is not
self-scored. Keep the phrasing families here distinct from the eval set.

Usage:
  python scripts/gen_rebate_synthetic.py --count 5200 --out datasets/synthetic/rebate_scam.jsonl
"""
import argparse
import json
import random
from pathlib import Path

SOURCE = "syn_rebate"
GENERATOR = "template-gen/minimax-M3.1"
CATEGORY = "rebate_scam"

# Real-world platforms / channels / payout words. Combined they explode the
# sample space while keeping every string a plausible Chinese scam SMS.
PLATFORMS = [
    "淘宝", "京东", "拼多多", "闲鱼", "抖音", "快手", "美团", "唯品会",
    "苏宁易购", "得物", "小红书", "微信", "QQ", "游戏",
]
CHANNELS = [
    "客服", "商家", "店家", "老板", "导师", "带单老师", "公司", "工作室",
]
TASKS = [
    "刷单", "刷好评", "刷销量", "刷信誉", "做任务", "做单", "拍单", "下单",
    "点赞", "关注公众号", "加购", "浏览商品", "抢单", "录入数据",
]
PAYOUTS = [
    "佣金", "返利", "返现", "提成", "返款", "红包", "补贴", "分成",
]
AMOUNTS = [5, 6, 8, 10, 12, 15, 18, 20, 25, 28, 30, 35, 40, 50, 60, 80, 100]
SETTLE = [
    "日结", "当天返", "当日到账", "即时返", "做完就返", "周结", "每单现结",
    "晚上统一打款", "24小时内到账",
]
PROMISE = [
    "月入过万", "轻松日入几百", "在家就能做", "时间自由", "无经验可做",
    "单子多不愁", "门槛低", "包教包会", "大量招收", "名额有限",
]
SEDUCE = [
    "诚招", "急招", "长期招", "现招募", "招兼职", "招聘", "找人手",
]
# Advance-fee / collateral variants (the dangerous 垫付 flow).
ADVANCE = [
    "先垫付本金", "需先垫资", "下单后先付保证金", "需充值激活",
    "先拍下指定商品", "需先付货款", "先交保证金", "先垫付订单款",
]
REFUND = [
    "做满{每单}单全额退还", "做完连本带利返", "完成全部订单返本金",
    "做满5单保证金退还", "当日全额返还",
]
# Benign-looking hooks that make the ad read like a real job post.
BENIGN_HOOK = [
    "无任何费用", "不收押金", "无需垫资", "只需提供银行卡", "纯手机操作",
    "在家轻松做", "不耽误本职工作", "男女不限", "学生党可做", "宝妈可做",
]

FILLER_OPEN = [
    "{platform}{task}，{每单}元{payout}，{settle}。",
    "诚招{platform}{task}员，{每单}元{payout}，{settle}。",
    "{platform}急招{task}，{每单}元{payout}，{settle}。",
    "招{platform}{task}兼职，{每单}元{payout}，{settle}。",
    "{platform}{task}长期招募，{每单}元{payout}，{settle}。",
    "寻找{platform}{task}帮手，{每单}元{payout}，{settle}。",
    "{platform}商家直招{task}，{每单}元{payout}，{settle}。",
    "招募{platform}{task}高手，{每单}元{payout}，{settle}。",
]
FILLER_MID = [
    "{promise}，{benign}，有意私聊。",
    "{promise}，{benign}，速报名。",
    "{promise}，名额仅限今日，{benign}。",
    "{promise}，{benign}，详情咨询{channel}。",
    "{advance}后{每单}元{payout}，{promise}。",
    "{advance}，{refund}，{settle}。",
    "{advance}再{task}，{每单}元{payout}，{settle}。",
    "{settle}，{promise}，{benign}，{speak}。",
]
FILLER_TAIL = [
    "{settle}，{benign}，{speak}。",
    "{promise}，{speak}，{settle}。",
    "{speak}，{promise}，{benign}。",
    "{benign}，{settle}，{speak}。",
    "{settle}，{promise}，速联系{channel}。",
    "{refund}，{speak}。",
]
SPEAK = [
    "有意请私聊", "速回", "详情私聊", "咨询加V", "点击链接报名",
    "详情咨询{channel}", "备注刷单进群", "加{channel}了解",
    "名额仅限今天", "先咨询后做单",
]


def _fill(tpl: str, rng: random.Random, ctx: dict) -> str:
    out = tpl
    for key in ("platform", "task", "payout", "settle", "promise", "benign",
                "channel", "advance", "refund", "speak", "每单"):
        if "{" + key + "}" in out:
            out = out.replace("{" + key + "}", str(ctx[key]))
    return out


# Slots whose repetition inside a single message reads as templated garbage.
_UNIQUE_SLOTS = ("settle", "promise", "benign", "advance", "refund", "speak", "每单")
_SLOT_TO_CTX = {
    "settle": "settle", "promise": "promise", "benign": "benign",
    "advance": "advance", "refund": "refund", "speak": "speak", "每单": "每单",
}


def _tpl_slots(tpl: str) -> set[str]:
    return {s for s in _UNIQUE_SLOTS if "{" + s + "}" in tpl}


def _reroll_if_dup(tpl: str, ctx: dict, used: set[str], rng: random.Random) -> str:
    """Re-draw any slot already used in this message so clauses don't repeat."""
    for slot in _tpl_slots(tpl):
        val = ctx[_SLOT_TO_CTX[slot]]
        if val in used:
            pool = {
                "settle": SETTLE, "promise": PROMISE, "benign": BENIGN_HOOK,
                "advance": ADVANCE, "speak": SPEAK,
            }.get(slot)
            if slot == "每单":
                alt = [a for a in AMOUNTS if str(a) not in used]
                if alt:
                    ctx["每单"] = rng.choice(alt)
            elif slot == "refund":
                alts = [r.replace("{每单}", str(a)) for r in REFUND
                        for a in [rng.choice(AMOUNTS)]]
                alts = [r for r in alts if r not in used]
                if alts:
                    ctx["refund"] = rng.choice(alts)
            elif pool:
                alts = [p for p in pool if p not in used]
                if alts:
                    ctx[_SLOT_TO_CTX[slot]] = rng.choice(alts)
    return tpl


def build_context(rng: random.Random) -> dict:
    return {
        "platform": rng.choice(PLATFORMS),
        "channel": rng.choice(CHANNELS),
        "task": rng.choice(TASKS),
        "payout": rng.choice(PAYOUTS),
        "每单": rng.choice(AMOUNTS),
        "settle": rng.choice(SETTLE),
        "promise": rng.choice(PROMISE),
        "advance": rng.choice(ADVANCE),
        "refund": rng.choice(REFUND).replace("{每单}", str(rng.choice(AMOUNTS))),
        "speak": rng.choice(SPEAK).replace("{channel}", rng.choice(CHANNELS)),
        "benign": rng.choice(BENIGN_HOOK),
    }


def build_text(rng: random.Random) -> str:
    ctx = build_context(rng)
    used: set[str] = set()
    open_tpl = rng.choice(FILLER_OPEN)
    used |= {ctx[s] for s in _tpl_slots(open_tpl)}
    parts = [_fill(open_tpl, rng, ctx)]
    n_mid = rng.choices([1, 2], weights=[0.55, 0.45])[0]
    for _ in range(n_mid):
        # Prefer mid templates whose slots are still unused.
        cands = [t for t in FILLER_MID
                 if not ({ctx[s] for s in _tpl_slots(t)} & used)]
        tpl = rng.choice(cands or FILLER_MID)
        tpl = _reroll_if_dup(tpl, ctx, used, rng)
        parts.append(_fill(tpl, rng, ctx))
        used |= {ctx[s] for s in _tpl_slots(tpl)}
    tail_cands = [t for t in FILLER_TAIL
                  if not ({ctx[s] for s in _tpl_slots(t)} & used)]
    tail_tpl = rng.choice(tail_cands or FILLER_TAIL)
    tail_tpl = _reroll_if_dup(tail_tpl, ctx, used, rng)
    parts.append(_fill(tail_tpl, rng, ctx))
    return "".join(parts).replace("名額", "名额")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=5200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="datasets/synthetic/rebate_scam.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seen: set[str] = set()
    rows: list[dict] = []
    tries = 0
    max_tries = args.count * 60
    while len(rows) < args.count and tries < max_tries:
        tries += 1
        text = build_text(rng).strip()
        if not (18 <= len(text) <= 70):
            continue
        if text in seen:
            continue
        seen.add(text)
        rows.append({
            "text": text,
            "is_scam": 1,
            "risk": 4,
            "category": CATEGORY,
            "language": "zh",
            "source": SOURCE,
            "generator": GENERATOR,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✓ {out}: {len(rows)} 条唯一 rebate_scam 正样本 (tries={tries})")
    lens = [len(r["text"]) for r in rows]
    print(f"  长度 min/max/avg = {min(lens)}/{max(lens)}/{sum(lens)/len(lens):.1f}")
    return 0 if len(rows) >= args.count else 1


if __name__ == "__main__":
    raise SystemExit(main())
