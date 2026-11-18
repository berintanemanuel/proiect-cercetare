"""Download and sample real CoCoLoFa benchmark items and load project datasets."""

from html import unescape
import json
from pathlib import Path
import random
import re
import urllib.request

COCOLOFA_TEST_URL = "https://raw.githubusercontent.com/Crowd-AI-Lab/cocolofa/main/test.json"

COCOLOFA_LABEL_MAP = {
    "appeal to authority": "Appeal to Authority",
    "appeal to majority": "Appeal to Majority",
    "appeal to nature": "Appeal to Nature",
    "appeal to tradition": "Appeal to Tradition",
    "appeal to worse problems": "Appeal to Worse Problems",
    "false dilemma": "False Dilemma",
    "hasty generalization": "Hasty Generalization",
    "slippery slope": "Slippery Slope",
    "none": "Valid",
}


def clean_html(raw_html: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    text = re.sub(r"<[^>]+>", " ", raw_html)
    text = unescape(text)
    return " ".join(text.split())


def build_cocolofa_sample(
    output_path: str = "data/cocolofa_sample.json",
    seed: int = 42,
    fallacy_per_class: int = 18,
    valid_count: int = 56,
) -> list[dict]:
    """Download CoCoLoFa test.json and sample 200 stratified instances (18 per fallacy + 56 Valid).

    Draws the original 100-item base sample first (11 per fallacy + 12 Valid) so existing
    cached evaluations remain identical, then appends deterministic extra samples.
    """
    req = urllib.request.Request(COCOLOFA_TEST_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        articles = json.loads(resp.read().decode("utf-8"))

    by_label: dict[str, list[dict]] = {label: [] for label in COCOLOFA_LABEL_MAP.values()}

    for art in articles:
        title = art.get("title", "").strip()
        body = clean_html(art.get("content", ""))
        if len(body) > 600:
            body = body[:600].rsplit(" ", 1)[0] + "..."

        comments_by_id = {
            str(c["id"]): c["comment"].strip() for c in art.get("comments", [])
        }

        for c in art.get("comments", []):
            raw_f = c.get("fallacy", "").strip().lower()
            if raw_f not in COCOLOFA_LABEL_MAP:
                continue

            canon_label = COCOLOFA_LABEL_MAP[raw_f]
            ctx = f'News Article: "{title}". Excerpt: {body}'
            resp_to = str(c.get("respond_to", "")).strip()
            if resp_to and resp_to in comments_by_id:
                ctx += f' | Replying to comment: "{comments_by_id[resp_to]}"'

            item = {
                "id": f"cocolofa_{c['id']}",
                "news_id": art["id"],
                "condition": "benchmark",
                "domain": "news_comments",
                "argument": c["comment"].strip(),
                "context": ctx,
                "true_label": canon_label,
                "raw_fallacy": raw_f,
            }
            by_label[canon_label].append(item)

    rng = random.Random(seed)
    sampled: list[dict] = []
    base_ids: set[str] = set()
    for label in sorted(by_label.keys()):
        base_k = 12 if label == "Valid" else 11
        chosen = rng.sample(by_label[label], base_k)
        sampled.extend(chosen)
        base_ids.update(item["id"] for item in chosen)

    for label in sorted(by_label.keys()):
        base_k = 12 if label == "Valid" else 11
        target_k = valid_count if label == "Valid" else fallacy_per_class
        extra_k = max(0, target_k - base_k)
        if extra_k > 0:
            remaining = [item for item in by_label[label] if item["id"] not in base_ids]
            sampled.extend(rng.sample(remaining, extra_k))

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(sampled, indent=2, ensure_ascii=False), encoding="utf-8")
    return sampled


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    sample_path = project_root / "data" / "cocolofa_sample.json"
    items = build_cocolofa_sample(str(sample_path))
    print(f"Saved {len(items)} real CoCoLoFa instances to {sample_path}")
