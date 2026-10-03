"""Mistral OCR: open-access PDF -> markdown full text (cached on disk)."""
import hashlib
from pathlib import Path

from dotenv import load_dotenv

from integrations.llm import MISTRAL_KEYS, post_with_failover

load_dotenv()

CACHE = Path(__file__).resolve().parents[1] / "data/papers"


def ocr(pdf_url, max_pages=12):
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / (hashlib.sha1(pdf_url.encode()).hexdigest() + ".md")
    if cached.exists():
        return cached.read_text()
    data = post_with_failover("https://api.mistral.ai/v1/ocr", MISTRAL_KEYS, {
        "model": "mistral-ocr-latest",
        "document": {"type": "document_url", "document_url": pdf_url},
        "pages": list(range(max_pages)),
    }, timeout=180)
    text = "\n".join(p["markdown"] for p in data["pages"])
    cached.write_text(text)
    return text
