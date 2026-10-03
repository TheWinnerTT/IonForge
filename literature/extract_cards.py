"""Paper text -> EvidenceCards (family-level trends only), using Claude Haiku via OpenRouter (or Anthropic)."""
import hashlib
import json
import re

from dotenv import load_dotenv

from integrations.llm import complete

load_dotenv()

SYSTEM = """You are a Literature Scout for an autonomous materials lab searching for lithium superionic solid electrolytes.
Extract design trends from the paper text that could guide which materials to test next.

Rules:
- Report only family-level trends (e.g. "In argyrodites, more Cl on the 4c site tends to raise conductivity").
- Never report a conductivity value for a specific composition, not even in the quote.
- The quote must be copied verbatim from the text (one or two sentences, max 300 characters). No paraphrasing.
- Prefer quotes that state a mechanism or direction of change, not a number.
- If the paper has no useful trend, return an empty list.

Return only JSON: {"cards": [{"family": str, "trend": str, "quote": str, "compositions_mentioned": [str]}]}"""

MAX_CHARS = 60_000


def extract_cards(paper, text, scout_family, n_max=4):
    user = (
        f"Scout family: {scout_family}\nTitle: {paper.get('title')}\nDOI: {paper.get('doi')}\n"
        f"Return at most {n_max} cards.\n\n--- PAPER TEXT ---\n{text[:MAX_CHARS]}"
    )
    raw = complete(SYSTEM, user, max_tokens=1500)
    match = re.search(r"\{.*\}", raw, re.S)
    cards = json.loads(match.group(0)).get("cards", []) if match else []
    out = []
    for c in cards[:n_max]:
        cid = hashlib.sha1(f"{paper.get('doi')}|{c.get('quote')}".encode()).hexdigest()[:16]
        out.append({
            "id": cid,
            "doi": paper.get("doi"),
            "title": paper.get("title"),
            "family": c.get("family") or scout_family,
            "scout": scout_family,
            "trend": c.get("trend", ""),
            "quote": c.get("quote", ""),
            "compositions_mentioned": c.get("compositions_mentioned", []),
            "source": paper.get("source"),
        })
    return out
