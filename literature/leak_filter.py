"""No-leak rule: literature may never hand the lab the oracle's answer.

A card is blocked when it mentions a pool composition (exact reduced formula, or a
doped variant of the same near-duplicate group) AND carries a conductivity value.
Blocked cards are stored with blocked_leak=true so they can be counted, but they
are never passed to the Hypothesis Generator.
"""
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
from pymatgen.core import Composition

POOL_CSV = Path(__file__).resolve().parents[1] / "data/pool.csv"
MAJOR_FRACTION = 0.04  # keep in sync with scripts/eda_obelix.py

# Conductivity-like values: 1.2 × 10−3, 1e-3, 10^-4, 12 mS/cm, 3.5 mS cm-1, log σ = -3
VALUE_RE = re.compile(
    r"(\d+(\.\d+)?\s*[x×*]\s*10\s*\^?\s*[-−–]\s*\d+)"
    r"|(\b\d+(\.\d+)?\s*e\s*[-−–]\s*\d+\b)"
    r"|(\b10\s*\^?\s*[-−–]\s*\d+\s*(S|mS))"
    r"|(\d+(\.\d+)?\s*[mµμ]?S\s*/?\s*cm)"
    r"|(log\s*\(?\s*[σs]\s*\)?\s*[=≈~]\s*[-−–]?\d)",
    re.I,
)
# Formula-like tokens containing Li, e.g. Li6PS5Cl, Li7La3Zr2O12, Li0.33La0.56TiO3
FORMULA_RE = re.compile(r"\b(?=[A-Za-z0-9.]*Li)(?:[A-Z][a-z]?(?:\d+(?:\.\d+)?)?){2,}\b")


def reduced(formula):
    try:
        return Composition(str(formula)).reduced_formula
    except Exception:
        return None


def host_system(formula):
    try:
        comp = Composition(str(formula)).fractional_composition
    except Exception:
        return None
    return "-".join(sorted(el.symbol for el, x in comp.items() if x >= MAJOR_FRACTION))


@lru_cache(maxsize=1)
def pool_index():
    pool = pd.read_csv(POOL_CSV)
    return set(pool["reduced"].dropna()), set(pool["host"].dropna())


def mentioned_compositions(card):
    found = set(card.get("compositions_mentioned") or [])
    for field in ("quote", "trend"):
        found.update(FORMULA_RE.findall(card.get(field) or ""))
    return sorted(found)


def leak_filter(card):
    pool_reduced, pool_hosts = pool_index()
    text = f"{card.get('trend', '')} {card.get('quote', '')}"
    hits = []
    for c in mentioned_compositions(card):
        r = reduced(c)
        if r is None:
            continue
        if r in pool_reduced:
            hits.append(f"{c} (pool material)")
        elif host_system(c) in pool_hosts:
            hits.append(f"{c} (doped variant of a pool group)")
    has_value = bool(VALUE_RE.search(text))
    card["blocked_leak"] = bool(hits) and has_value
    card["blocked_reason"] = f"conductivity value for {', '.join(hits)}" if card["blocked_leak"] else None
    return card


if __name__ == "__main__":
    tests = [
        {"trend": "In argyrodites, more Cl on the 4c site tends to raise conductivity.", "quote": "increasing Cl content enhances Li-ion conductivity"},
        {"trend": "Li6PS5Cl reaches 1.9 mS/cm.", "quote": "Li6PS5Cl shows 1.9 × 10−3 S cm−1 at room temperature"},
        {"trend": "Ta-doped LLZO is highly conductive.", "quote": "Li6.4La3Zr1.4Ta0.6O12 exhibits 1e-3 S/cm"},
    ]
    for t in tests:
        print(leak_filter(t)["blocked_leak"], t["blocked_reason"])
