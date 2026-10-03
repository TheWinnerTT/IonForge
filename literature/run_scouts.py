"""Run the three Literature Scouts in parallel and store EvidenceCards.

Pipeline per family: search (OpenAlex + arXiv [+ BrightData]) -> full text
(Mistral OCR, fallback to abstract) -> Mistral extraction -> citation check
-> no-leak filter -> results/evidence_cards.json + Supabase `evidence`.

Usage:
  python -m literature.run_scouts                 # all families
  python -m literature.run_scouts --papers 4 --families halides
  python -m literature.run_scouts --brightdata-only --papers 9   # top up the cache
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
from literature.openalex import QUERIES, search_family, work_by_doi
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


def brightdata_papers(family, known_dois, limit):
    """Papers only BrightData (Google Scholar) found: DOI new to OpenAlex/arXiv search,
    text (abstract / open-access PDF) fetched from OpenAlex by DOI."""
    out = []
    for hit in serp(QUERIES[family][0], n=10):
        doi = (hit.get("doi") or "").lower()
        if not doi or doi in known_dois:
            continue
        work = work_by_doi(doi)
        if not work or not (work["abstract"] or work["pdf"]):
            continue
        out.append({**hit, **{k: v for k, v in work.items() if v}, "doi": doi, "family": family, "source": "brightdata"})
        known_dois.add(doi)
        if len(out) >= limit:
            break
    return out


def read_papers(papers, family):
    cards = []
    for p in papers:
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
        print(f"  [{family}] {p['doi']}: {len(new)} cards ({kind}, {p.get('source')})")
    return cards


def add_brightdata_cards(families, per_family=3):
    """Append cards from BrightData-only papers to the shared cache without re-reading
    the papers already there (cheap top-up for an existing evidence store)."""
    cache = json.loads(OUT.read_text()) if OUT.exists() else []
    known = {(c.get("doi") or "").lower() for c in cache}
    added = []
    for family in families:
        known |= {p["doi"].lower() for p in search_family(family) if p.get("doi")}
        for c in read_papers(brightdata_papers(family, known, per_family), family):
            c["scout"] = family
            added.append(c)
    ids = {c["id"] for c in cache}
    new = [c for c in added if c["id"] not in ids]
    OUT.write_text(json.dumps(cache + new, indent=2, ensure_ascii=False))
    if new:
        db.upsert("evidence", [{k: c.get(k) for k in DB_FIELDS} for c in new])
    passed = [c for c in new if not c["blocked_leak"]]
    print(f"\nBrightData: {len(new)} new cards | {len(passed)} passed | "
          f"{sum(c['verified'] for c in passed)} verified -> {OUT}")
    return new


def scout(family, n_papers, use_brightdata):
    papers = [p for p in search_family(family) if p.get("doi")]
    # Prefer papers we can read in full.
    papers.sort(key=lambda p: (p.get("pdf") is None, -(p.get("year") or 0)))
    papers = papers[:n_papers]
    if use_brightdata:
        # Reserve a third of the slots for papers the open indexes did not surface.
        try:
            extra = brightdata_papers(family, {p["doi"].lower() for p in papers}, limit=max(1, n_papers // 3))
        except Exception as e:
            print(f"  [{family}] BrightData unavailable: {e}")
            extra = []
        papers = papers[:n_papers - len(extra)] + extra
        print(f"  [{family}] {len(extra)} papers found only by BrightData")
    return read_papers(papers, family)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", type=int, default=8, help="papers per family")
    ap.add_argument("--families", nargs="*", default=list(QUERIES))
    ap.add_argument("--brightdata", action="store_true")
    ap.add_argument("--brightdata-only", action="store_true",
                    help="only add cards from papers BrightData found, keep the existing cache")
    args = ap.parse_args()
    if args.brightdata_only:
        add_brightdata_cards(args.families, per_family=max(1, args.papers // 3))
        return

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
