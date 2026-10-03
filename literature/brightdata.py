"""BrightData SERP API: finds papers/preprints OpenAlex does not cover.

Only search result pages are fetched (titles + links); full texts still come from
open-access PDFs. Respect each site's terms of use.
"""
import json
import os
import re
import urllib.parse

import requests
from dotenv import load_dotenv

load_dotenv()

DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>?#]+", re.I)


NATURE_RE = re.compile(r"nature\.com/articles/([a-z0-9.\-]+)", re.I)


def _unwrap(link):
    """Google Scholar redirect links carry the real URL in ?url=."""
    q = urllib.parse.parse_qs(urllib.parse.urlparse(link).query)
    return q["url"][0] if "scholar_url" in link and "url" in q else link


def _doi(link, text):
    m = DOI_RE.search(f"{link} {text}")
    if m:
        return re.sub(r"/(meta|pdf|full|abstract|epdf)$", "", m.group(0).rstrip("."), flags=re.I)
    m = NATURE_RE.search(link)
    return f"10.1038/{m.group(1)}" if m else None


def serp(query, n=10):
    key = os.getenv("BRIGHTDATA_API_KEY")
    if not key:
        return []
    url = f"https://www.google.com/search?q={urllib.parse.quote_plus(query)}"
    r = requests.post(
        "https://api.brightdata.com/request",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"zone": os.getenv("BRIGHTDATA_SERP_ZONE", "ionforge"), "url": url,
              "format": "json", "data_format": "parsed"},
        timeout=60,
    )
    r.raise_for_status()
    resp = r.json()
    if resp.get("status_code", 200) != 200:
        print(f"  BrightData: {resp.get('status_code')} {str(resp.get('body'))[:120]}")
        return []
    body = resp.get("body", {})
    data = json.loads(body) if isinstance(body, str) else body
    scholar = data.get("scholarly_articles") or {}
    items = (scholar.get("items", []) if isinstance(scholar, dict) else scholar) + (data.get("organic") or [])
    out = []
    for item in items:
        link = _unwrap(item.get("link") or "")
        out.append({
            "doi": _doi(link, item.get("description", "")),
            "title": item.get("title"),
            "year": None,
            "pdf": link if link.lower().endswith(".pdf") else None,
            "abstract": item.get("description", ""),
            "url": link,
            "source": "brightdata",
        })
    return out[:n]


if __name__ == "__main__":
    for p in serp("argyrodite solid electrolyte halogen disorder lithium conductivity", n=5):
        print(p["doi"], p["title"])
