"""BrightData SERP API: finds papers/preprints OpenAlex does not cover.

Only search result pages are fetched (titles + links); full texts still come from
open-access PDFs. Respect each site's terms of use.
"""
import os
import re
import urllib.parse

import requests
from dotenv import load_dotenv

load_dotenv()

DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>?#]+", re.I)


def serp(query, engine="scholar", n=10):
    key = os.getenv("BRIGHTDATA_API_KEY")
    if not key:
        return []
    base = "https://scholar.google.com/scholar" if engine == "scholar" else "https://www.google.com/search"
    url = f"{base}?q={urllib.parse.quote_plus(query)}&num={n}&brd_json=1"
    r = requests.post(
        "https://api.brightdata.com/request",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"zone": os.getenv("BRIGHTDATA_SERP_ZONE", "serp_api1"), "url": url, "format": "raw"},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    out = []
    for item in data.get("organic", [])[:n]:
        link = item.get("link") or ""
        doi = DOI_RE.search(link + " " + item.get("description", ""))
        out.append({
            "doi": doi.group(0).rstrip(".") if doi else None,
            "title": item.get("title"),
            "year": None,
            "pdf": link if link.lower().endswith(".pdf") else None,
            "abstract": item.get("description", ""),
            "url": link,
            "source": "brightdata",
        })
    return out


if __name__ == "__main__":
    for p in serp("lithium halide solid electrolyte conductivity trend", n=5):
        print(p["doi"], p["title"])
