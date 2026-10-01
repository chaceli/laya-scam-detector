"""arkcli 多模型生成难负样本（design §6.1 八体裁 + §6.2 对照对）。

多模型轮转避免单一文体；每行记录 generator。QC：长度 10-160、含 CJK、全库去重。
用法：
  python scripts/generate_negatives.py --genre hn_financial_notice --limit 25  # pilot
  python scripts/generate_negatives.py                                        # 全量
  python scripts/generate_negatives.py --genre hn_promotion --append          # 补量
  python scripts/generate_negatives.py --out datasets/hard_negatives/mining_candidates.jsonl  # 挖掘候选池
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_mix import stable_id

DATA = Path("datasets/hard_negatives")
CFG = Path("scripts/gen_models.json")
BATCH = 25

PROMPT_TMPL = """你在为中文诈骗检测模型构造「难负样本」：看起来像诈骗、实际完全合法、模型容易误报的文本。

体裁：{desc}
内容要素：{elements}
严禁出现：{must_not}

要求：
1. 每行输出一条 JSON：{{"text": "..."}}，共 {n} 条，不要编号、不要解释、不要 markdown 代码块
2. 文本 20-120 字，像真实中文短信，语气自然
3. 多样化：机构名/人名/长度/正式度/标点都要变化，禁止重复句式
4. 直接输出 JSONL，第一行开始就是 JSON"""

PAIR_TMPL = """对照对构造：生成 {pairs} 对短信，每对一合法一诈骗。

体裁：{desc}
合法版内容要素：{elements}
合法版严禁：{must_not}
两版结构、语气、署名、句式完全一致，唯一差异：诈骗版{scam_delta}

输出格式（每对两行，共 {pairs} 对，不要编号不要解释）：
{{"is_scam": 0, "text": "合法版内容"}}
{{"is_scam": 1, "text": "诈骗版内容"}}"""

GENRES = {
    "hn_financial_notice": {
        "target": 1200,
        "desc": "银行/券商/保险/支付机构发给客户的真实业务通知短信",
        "elements": "机构署名（如【XX银行】【XX证券】）、验证码、额度调整、账单出账、积分到期、密码重置成功",
        "must_not": "让用户把验证码告诉任何人、引导添加私人微信QQ好友、非官方域名链接、索要密码",
    },
    "hn_ecommerce_logistics": {
        "target": 1200,
        "desc": "电商平台/快递公司的正常通知短信",
        "elements": "订单状态、发货、签收、取件码、地址变更确认、退货进度",
        "must_not": "索费、引导点击非官方链接登录、加私人好友",
    },
    "hn_job_ad": {
        "target": 800,
        "desc": "正规公司招聘/兼职广告短信",
        "elements": "岗位名称、薪资范围、面试地点、公司名",
        "must_not": "预付费用、押金、境外高薪、刷单",
    },
    "hn_promotion": {
        "target": 1000,
        "desc": "商家促销/会员权益通知短信",
        "elements": "满减、折扣、积分兑换、会员日、优惠券到账",
        "must_not": "要求先转账再领奖、垫资、缴纳保证金",
    },
    "hn_gov_notice": {
        "target": 1200,
        "desc": "政务/公共机构通知短信",
        "elements": "社保、医保、公积金、ETC、违章提醒、学校家长通知、单位会议、天气预警",
        "must_not": "索要密码验证码、索费、引导转账",
    },
    "hn_personal_social": {
        "target": 1200,
        "desc": "家人朋友之间的日常短信",
        "elements": "问候、约饭、拼团、红包、代付提醒、到家叮嘱",
        "must_not": "任何机构署名、索费",
    },
    "hn_traffic_funnel": {
        "target": 1000,
        "desc": "合法引流/内容推广短信",
        "elements": "交友活动邀请、直播预告、内容更新提醒、公众号推广",
        "must_not": "索钱、投资引导、刷单",
    },
    "nb_natural_benign": {
        "target": 1500,
        "desc": "最普通的日常短信",
        "elements": "快递取件码、验证码、会议通知、家人问候、缴费提醒",
        "must_not": "任何诈骗结构或索费",
    },
}

CONTRASTIVE_GROUPS = [
    {"genre": "hn_financial_notice", "scam_category": "impersonation",
     "scam_delta": "以银行客服口吻要求提供验证码或点击钓鱼链接办理业务"},
    {"genre": "hn_ecommerce_logistics", "scam_category": "delivery_fraud",
     "scam_delta": "声称包裹丢失需点击链接填写银行卡信息理赔"},
    {"genre": "hn_job_ad", "scam_category": "job_scam",
     "scam_delta": "要求先缴报名费押金培训费"},
]
PAIRS_PER_GROUP = 100


def call_model(cfg: dict, model: str, prompt: str) -> str:
    argv = [a.replace("{model}", model).replace("{prompt}", prompt)
            for a in cfg["argv_template"]]
    r = subprocess.run(argv, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"exit {r.returncode}: {r.stderr[:200]}")
    return r.stdout


def parse_rows(out: str) -> list[dict]:
    rows = []
    for line in out.splitlines():
        line = line.strip().strip("`")
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict) and isinstance(rec.get("text"), str):
            rows.append(rec)
    return rows


def qc(text: str) -> bool:
    return 10 <= len(text) <= 160 and any("\u4e00" <= c <= "\u9fff" for c in text)


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("a" if path.exists() else "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_existing_texts(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(l)["text"] for l in path.open() if l.strip()}


def gen_genre(cfg: dict, name: str, spec: dict, limit: int | None,
              seen: set[str], out_override: str | None = None) -> int:
    out_path = Path(out_override) if out_override else DATA / f"{name}.jsonl"
    seen |= load_existing_texts(out_path)
    need = min(spec["target"], limit) if limit else spec["target"]
    fresh = 0
    call = 0
    while fresh < need and call < (need // BATCH) * 3 + 6:
        model = cfg["models"][call % len(cfg["models"])]
        prompt = PROMPT_TMPL.format(
            desc=spec["desc"], elements=spec["elements"],
            must_not=spec["must_not"], n=min(BATCH, need - fresh))
        try:
            out = call_model(cfg, model, prompt)
        except Exception as e:
            print(f"  ! {name} model {model} 失败，换下一个: {e}")
            call += 1
            time.sleep(2)
            continue
        rows = []
        for rec in parse_rows(out):
            t = rec["text"].strip()
            if qc(t) and t not in seen:
                seen.add(t)
                rows.append({
                    "id": stable_id(t, name), "text": t, "is_scam": 0,
                    "risk": 1, "category": "benign", "language": "zh",
                    "source": name, "generator": model,
                })
                fresh += 1
        write_rows(out_path, rows)
        call += 1
        print(f"  {name}: call {call} (+{len(rows)}, total fresh {fresh}/{need})")
    return fresh


def gen_contrastive(cfg: dict, seen: set[str]) -> int:
    out_path = DATA / "contrastive_pairs.jsonl"
    seen |= load_existing_texts(out_path)
    total = 0
    for g in CONTRASTIVE_GROUPS:
        spec = GENRES[g["genre"]]
        pairs_done = 0
        call = 0
        while pairs_done < PAIRS_PER_GROUP and call < PAIRS_PER_GROUP * 2 + 4:
            model = cfg["models"][call % len(cfg["models"])]
            prompt = PAIR_TMPL.format(
                pairs=min(BATCH, PAIRS_PER_GROUP - pairs_done),
                desc=spec["desc"], elements=spec["elements"],
                must_not=spec["must_not"], scam_delta=g["scam_delta"])
            try:
                out = call_model(cfg, model, prompt)
            except Exception as e:
                print(f"  ! contrastive {g['genre']} 失败: {e}")
                call += 1
                time.sleep(2)
                continue
            rows = []
            for rec in parse_rows(out):
                t = rec["text"].strip()
                is_scam = int(rec.get("is_scam", -1))
                if not qc(t) or t in seen or is_scam not in (0, 1):
                    continue
                seen.add(t)
                rows.append({
                    "id": stable_id(t, "contrastive_pair"), "text": t,
                    "is_scam": is_scam, "risk": 4 if is_scam else 1,
                    "category": g["scam_category"] if is_scam else "benign",
                    "language": "zh", "source": "contrastive_pair",
                    "generator": model,
                })
                if is_scam == 1:
                    pairs_done += 1
            write_rows(out_path, rows)
            total += len(rows)
            call += 1
            print(f"  contrastive {g['genre']}: call {call} (+{len(rows)})")
    return total


def main() -> int:
    if not CFG.exists():
        print("✗ scripts/gen_models.json 缺失 —— 先完成 Step 1 探测")
        return 2
    cfg = json.loads(CFG.read_text())
    if len(cfg.get("models", [])) < 3:
        print("✗ gen_models.json 需 ≥3 个模型（多模型分散文体）")
        return 2

    ap = argparse.ArgumentParser()
    ap.add_argument("--genre", help="只跑指定体裁（pilot 用）")
    ap.add_argument("--limit", type=int, help="每体裁上限（pilot 用）")
    ap.add_argument("--out", help="输出到指定文件（挖掘候选池），而非按体裁分文件")
    args = ap.parse_args()

    seen: set[str] = set()
    targets = {k: v for k, v in GENRES.items()
               if not args.genre or k == args.genre}
    for name, spec in targets.items():
        got = gen_genre(cfg, name, spec, args.limit, seen, args.out)
        print(f"✓ {name}: {got} 条")
    if not args.genre and not args.out:
        got = gen_contrastive(cfg, seen)
        print(f"✓ contrastive_pairs: {got} 行")
    return 0


if __name__ == "__main__":
    sys.exit(main())