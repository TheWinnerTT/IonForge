"""Citation validator: a card is verified only if its quote appears verbatim in the source text."""
import re
import unicodedata


def norm(s):
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = re.sub(r"[‐-―−]", "-", s)       # unify dashes
    s = re.sub(r"[“”\"'‘’`]", "", s)   # drop quotes
    s = re.sub(r"-\s*\n\s*", "", s)      # re-join hyphenated line breaks
    return re.sub(r"\s+", " ", s).strip()


def verify(card, fulltext):
    card["verified"] = bool(card.get("quote")) and bool(card.get("doi")) and norm(card["quote"]) in norm(fulltext)
    return card
