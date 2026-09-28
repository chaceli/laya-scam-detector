"""Interactive Laya scam-phrase test tool.

Run with:
    .venv/bin/python test_interactive.py
"""
import json
import sys

from src.router import Router

CASES = [
    ("zh", "中文公检法诈骗",
     "【最高人民检察院】您涉嫌洗钱，已立案侦查，请将资金转入安全账户配合调查，联系警官00861-396-xxx。"),
    ("zh", "中文快递理赔诈骗",
     "您好，我是XX快递客服，您有一个包裹在运输途中丢失，现需要您配合操作进行理赔。"),
    ("zh", "中文杀猪盘",
     "宝贝儿，我在这边投资了一个平台，稳赚不赔，你要不要和我一起？先投5000试试。"),
    ("en", "PayPal phishing",
     "URGENT: Your PayPal account has been limited. Click here to verify: http://paypa1-secure.tk/login"),
    ("en", "crypto investment scam",
     "Make $5000/day with our secret crypto trading algorithm. 100% guaranteed returns."),
    ("zh", "中文良性·朋友问候",
     "妈，我今晚回家吃饭，大概6点到家。"),
    ("zh", "中文良性·会议提醒",
     "老板，明天上午9点的会议改到下午3点，麻烦您准时参加。"),
    ("zh", "中文良性·快递签收",
     "您的快递已签收，感谢您选择京东，期待再次为您服务。"),
    ("en", "English benign·meeting",
     "Hi, just confirming our meeting at 3pm tomorrow. Looking forward to it."),
    ("en", "English benign·shipping",
     "Your Amazon order #12345 has shipped and will be delivered on Friday."),
]


def show_result(label: str, result: dict) -> None:
    a = result["answers"]
    noul = a["is_scam"]["noul"]
    score = a["risk_level"]["score"]
    cat = a["scam_category"]["choice"]
    top3 = sorted(a["scam_category"]["probabilities"].items(),
                  key=lambda x: -x[1])[:3]
    judgment = "🚨 诈骗" if noul >= 0.5 else "✅ 正常"
    print(f"  🛡  is_scam:    {noul:.3f}  ({judgment})")
    print(f"  📊  risk_level: {score:.2f} / {a['risk_level']['max_score']}")
    print(f"  🏷  category:   {cat}")
    print(f"     top-3: " + ", ".join(f"{k}={v:.2f}" for k, v in top3))
    print(f"  🛣  routing:    {result['routing']['model']}")
    print(f"  ⏱  latency:     {result['latency_ms']:.0f}ms")


def main() -> int:
    print("=" * 70)
    print("Laya 诈骗话术测试 — 交互模式")
    print("=" * 70)
    print("\n加载模型（首次约 1-2 秒）...")
    router = Router(
        english_dir="models/laya-onnx-en",
        multilingual_dir="models/laya-onnx-multilingual",
    )
    schema = json.load(open("schemas/scam.json"))
    print("✓ 模型加载完成\n")

    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        print("─── 批量运行内置测试用例 ───\n")
        for i, (lang, label, text) in enumerate(CASES, 1):
            print(f"[{i:2}/{len(CASES)}] {label}")
            result = router.predict(text, schema)
            show_result(label, result)
            print()
        return 0

    print("─── 输入模式 ───")
    print("支持三种用法：")
    print("  1) 直接输入文本（自动检测中英文，多行以 Ctrl-D 结束）")
    print("  2) --batch 跑内置 10 条典型用例")
    print("  3) --route <text> 只检测脚本不推理")
    print()

    while True:
        try:
            print("─" * 70)
            line = input("\n📝 请输入文本（Ctrl-D 退出；--batch 跑批量；--route <text> 仅路由）：\n> ")
            text = line.strip()
            if not text:
                continue
            if text == "--batch":
                for i, (lang, label, t) in enumerate(CASES, 1):
                    print(f"\n[{i:2}/{len(CASES)}] {label}")
                    result = router.predict(t, schema)
                    show_result(label, result)
                continue
            if text.startswith("--route "):
                from src.router import ScriptRouter
                sr = ScriptRouter()
                target, reason = sr.route_with_reason(text[len("--route "):].strip())
                print(f"  routed_to: {target}\n  reason: {reason}")
                continue

            result = router.predict(text, schema)
            show_result(text, result)
        except EOFError:
            print("\n\n退出。")
            return 0
        except KeyboardInterrupt:
            print("\n\n退出。")
            return 0


if __name__ == "__main__":
    sys.exit(main())