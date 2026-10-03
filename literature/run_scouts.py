"""Run the three Literature Scouts in parallel and store EvidenceCards.

Pipeline per family: search (OpenAlex + arXiv [+ BrightData]) -> full text
(Mistral OCR, fallback to abstract) -> Claude extraction -> citation check
-> no-leak filter -> results/evidence_cards.json + Supabase `evidence`.

Usage:
  python -m literature.run_scouts                 # all families
  python -m literature.run_scouts --papers 4 --families halides
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from literature.brightdata import serp
from literature.extract_cards import extract_cards
from literature.leak_filter import leak_filter
from literature.mistral_ocr import ocr
from literature.openalex import QUERIES, search_family
from literature.verify import verify
from integrations.supabase_sync import db

OUT = Path(__file__).resolve().parents[1] / "results/evidence_cards.json"
DB_FIELDS = ("id", "doi", "title", "family", "trend", "quote", "verified", "blocked_leak", "blocked_reason", "source")


def full_text(paper):
    if paper.get("pdf") and os.getenv("MISTRAL_API_KEY"):
        try:
            return ocr(paper["pdf"]), "fulltext"
        except Exception as e:
            print(f"  OCR failed for {paper['doi']}: {e}")
    return paper.get("abstract") or "", "abstract"


def scout(family, n_papers, use_brightdata):
    papers = search_family(family)
    if use_brightdata:
        papers += [{**p, "family": family} for p in serp(QUERIES[family][0], n=5)]
    papers = [p for p in papers if p.get("doi")]
    # Prefer papers we can read in full.
    papers.sort(key=lambda p: (p.get("pdf") is None, -(p.get("year") or 0)))
    cards = []
    for p in papers[:n_papers]:
        text, kind = full_text(p)
        if len(text) < 300:
            continue
        try:
            new = extract_cards(p, text, family)
        except Exception as e:
            print(f"  [{family}] extraction failed for {p['doi']}: {e}")
            continue
        for c in new:
            c["text_kind"] = kind
            cards.append(leak_filter(verify(c, text)))
        print(f"  [{family}] {p['doi']}: {len(new)} cards ({kind})")
    return cards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", type=int, default=8, help="papers per family")
    ap.add_argument("--families", nargs="*", default=list(QUERIES))
    ap.add_argument("--brightdata", action="store_true")
    args = ap.parse_args()

    with ThreadPoolExecutor(max_workers=len(args.families)) as pool:
        results = pool.map(lambda f: scout(f, args.papers, args.brightdata), args.families)
    cards = [c for batch in results for c in batch]

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(cards, indent=2, ensure_ascii=False))
    if cards:
        db.upsert("evidence", [{k: c.get(k) for k in DB_FIELDS} for c in cards])

    passed = [c for c in cards if not c["blocked_leak"]]
    print(f"\n{len(cards)} cards | {len(passed)} passed | "
          f"{len(cards) - len(passed)} blocked by the no-leak rule | "
          f"{sum(c['verified'] for c in passed)} verified quotes -> {OUT}")


def usable_cards(path=OUT):
    """Cards the Hypothesis Generator may see: not blocked and quote verified."""
    cards = json.loads(Path(path).read_text())
    return [c for c in cards if c["verified"] and not c["blocked_leak"]]


if __name__ == "__main__":
    main()
