"""Mistral OCR: open-access PDF -> markdown full text (cached on disk)."""
import hashlib
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

CACHE = Path(__file__).resolve().parents[1] / "data/papers"


def ocr(pdf_url, max_pages=12):
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / (hashlib.sha1(pdf_url.encode()).hexdigest() + ".md")
    if cached.exists():
        return cached.read_text()
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        raise RuntimeError("MISTRAL_API_KEY is not set")
    r = requests.post(
        "https://api.mistral.ai/v1/ocr",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "mistral-ocr-latest",
            "document": {"type": "document_url", "document_url": pdf_url},
            "pages": list(range(max_pages)),
        },
        timeout=180,
    )
    r.raise_for_status()
    text = "\n".join(p["markdown"] for p in r.json()["pages"])
    cached.write_text(text)
    return text
