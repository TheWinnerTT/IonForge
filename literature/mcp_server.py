"""Literature tools for the three Literature Scouts, served to Omnigent over MCP (stdio).

Wraps the evidence pipeline in literature/run_scouts.py:
search (OpenAlex + arXiv [+ BrightData]) -> full text (Mistral OCR, fallback abstract)
-> card extraction -> verbatim-quote check -> no-leak filter -> Supabase `evidence`.

Cards are cached in results/evidence_cards.json, so a family is read once and every
campaign seed reuses the same evidence (and the same cost). The Hypothesis Generator
only ever sees cards that are verified and not blocked by the no-leak rule.

Never print to stdout here: it is the MCP protocol channel.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys

from mcp.server.mcpserver import MCPServer

from literature.openalex import QUERIES

mcp = MCPServer("ionforge-literature",
                instructions="Evidence cards (family-level trends with verbatim, DOI-cited quotes) "
                             "for Li-ion solid electrolytes. Cards that would leak a pool material's "
                             "conductivity are blocked and only counted.")
CARDS = __import__("pathlib").Path(__file__).resolve().parents[1] / "results" / "evidence_cards.json"
PUBLIC_FIELDS = ("id", "family", "trend", "quote", "doi", "title", "year", "text_kind")


def _load() -> list[dict]:
    return json.loads(CARDS.read_text()) if CARDS.exists() else []


def _save(cards: list[dict]) -> None:
    CARDS.parent.mkdir(parents=True, exist_ok=True)
    CARDS.write_text(json.dumps(cards, indent=2, ensure_ascii=False))


def _report(family: str, cards: list[dict]) -> dict:
    mine = [c for c in cards if c.get("scout") == family]
    usable = [c for c in mine if c.get("verified") and not c.get("blocked_leak")]
    return {
        "family": family,
        "cards": [{k: c.get(k) for k in PUBLIC_FIELDS} for c in usable],
        "n_total": len(mine),
        "n_usable": len(usable),
        "n_blocked_by_no_leak_rule": sum(bool(c.get("blocked_leak")) for c in mine),
        "n_unverified_quote": sum(not c.get("verified") and not c.get("blocked_leak") for c in mine),
    }


@mcp.tool()
def scout_family(family: str, n_papers: int = 6, refresh: bool = False) -> dict:
    """Gather evidence cards for one family: "sulfides", "oxides" or "halides".

    Returns only usable cards (verbatim quote verified, not blocked by the no-leak rule)
    plus counts of blocked and unverified cards. Uses the cache unless refresh=True.
    """
    if family not in QUERIES:
        return {"error": f"family must be one of {sorted(QUERIES)}"}
    cards = _load()
    if refresh or not any(c.get("scout") == family for c in cards):
        from integrations.supabase_sync import db
        from literature.run_scouts import DB_FIELDS, scout

        with contextlib.redirect_stdout(sys.stderr):
            new = scout(family, max(1, min(n_papers, 12)), use_brightdata=bool(os.getenv("BRIGHTDATA_API_KEY")))
            cards = [c for c in cards if c.get("scout") != family] + new
            _save(cards)
            if new and db.enabled:
                db.upsert("evidence", [{k: c.get(k) for k in DB_FIELDS} for c in new])
    return _report(family, cards)


@mcp.tool()
def evidence_cards(family: str | None = None) -> list[dict]:
    """Usable cards already gathered (all families, or one). Never includes blocked cards."""
    out = []
    for fam in ([family] if family else sorted(QUERIES)):
        out += _report(fam, _load())["cards"]
    return out


if __name__ == "__main__":
    mcp.run()
