"""抓取公安/政府/运营商反诈宣传文本（难负样本体裁 1）。

URL 来自 dataset_research.md 引文；单 URL 失败优雅跳过（种子语料兜底）。
用法：python scripts/scrape_antifraud.py
"""
import json
import re
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_mix import stable_id

SEED = Path("datasets/hard_negatives/seed_propaganda.jsonl")
RAW_PAGES = Path("datasets/raw/propaganda")
OUT = Path("datasets/hard_negatives/hn_antifraud_propaganda.jsonl")
SOURCE = "hn_antifraud_propaganda"

URLS = [
    # 晋城公安反诈语录「十个凡是」（report 【23】）
    "http://ywtb.gaj.jcgov.gov.cn/site/public/showinfo.aspx?id=2026030609245557470105",
    # 平坝区致群众一封信（report 【29】）
    "https://www.pingba.gov.cn/xzjd/tlz/zfxxgk_5668073/fdzdgknr_5668076/xxgk/202607/t20260701_90572939.html",
    # 临汾移动 12381/96110 误区澄清（report 【24】）
    "https://lf.sxgov.cn/content/2026-08/25/content_13676534.htm",
    # 公安部 2026 宣传手册报道（report 【27】）
    "https://news.ycwb.com/ikimvkmtjj/content_54169914.htm",
]

KEYWORD = re.compile(r"反诈|诈骗|预警|劝阻|96110|12381|凡是")


class TextExtract(HTMLParser):
    SKIP = {"script", "style", "nav", "header", "footer"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip > 0:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip and data.strip():
            self.parts.append(data.strip())


def fetch(url: str) -> str:
    r = subprocess.run(["curl", "-sL", "-m", "60", url],
                       capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"curl exit {r.returncode}")
    return r.stdout


def extract_texts(html: str) -> list[str]:
    p = TextExtract()
    p.feed(html)
    return [t for t in p.parts if 15 <= len(t) <= 200 and KEYWORD.search(t)]


def main() -> int:
    RAW_PAGES.mkdir(parents=True, exist_ok=True)
    texts: list[str] = []
    with SEED.open() as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(json.loads(line)["text"])
    print(f"  ✓ seed: {len(texts)}")

    for i, url in enumerate(URLS):
        stem = f"page{i}"
        page = RAW_PAGES / f"{stem}.txt"
        try:
            html = fetch(url)
            page.write_text(html)
            got = extract_texts(html)
            texts.extend(got)
            print(f"  ✓ {url[:60]}... -> {len(got)} 段")
        except Exception as e:
            print(f"  ✗ {url[:60]}... 跳过: {e}")

    seen, rows = set(), []
    for t in texts:
        if t in seen:
            continue
        seen.add(t)
        rows.append({
            "id": stable_id(t, SOURCE), "text": t, "is_scam": 0, "risk": 1,
            "category": "benign", "language": "zh", "source": SOURCE,
        })
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✓ {OUT}: {len(rows)} 条（seed 20 + 抓取）")
    return 0


if __name__ == "__main__":
    sys.exit(main())