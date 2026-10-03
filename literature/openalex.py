"""Paper search: OpenAlex (open-access works) and arXiv."""
import os
import xml.etree.ElementTree as ET

import requests
from dotenv import load_dotenv

load_dotenv()

# One query list per Literature Scout (family-level, never a single pool material on purpose).
QUERIES = {
    "sulfides": [
        "argyrodite lithium solid electrolyte halogen substitution ionic conductivity",
        "thiophosphate lithium superionic conductor design",
        "LGPS-type sulfide electrolyte structure conductivity trend",
    ],
    "oxides": [
        "garnet lithium solid electrolyte doping cubic phase stabilization",
        "NASICON lithium conductor substitution bottleneck size",
        "perovskite lithium lanthanum titanate conductivity mechanism",
    ],
    "halides": [
        "halide solid electrolyte lithium chloride conductivity design",
        "lithium metal halide superionic conductor anion framework",
    ],
}


def search_openalex(query, n=25):
    params = {"search": query, "filter": "is_oa:true", "per-page": n}
    mailto = os.getenv("OPENALEX_MAILTO")
    if mailto:
        params["mailto"] = mailto
    r = requests.get("https://api.openalex.org/works", params=params, timeout=30)
    r.raise_for_status()
    out = []
    for w in r.json()["results"]:
        loc = w.get("best_oa_location") or {}
        out.append({
            "doi": (w.get("doi") or "").replace("https://doi.org/", "") or None,
            "title": w.get("title"),
            "year": w.get("publication_year"),
            "pdf": loc.get("pdf_url"),
            "abstract": _abstract(w.get("abstract_inverted_index")),
            "source": "openalex",
        })
    return out


def work_by_doi(doi):
    """OpenAlex record for one DOI (abstract + open-access PDF), or None if unknown."""
    params = {"mailto": os.getenv("OPENALEX_MAILTO")} if os.getenv("OPENALEX_MAILTO") else {}
    r = requests.get(f"https://api.openalex.org/works/doi:{doi}", params=params, timeout=30)
    if r.status_code != 200:
        return None
    w = r.json()
    loc = w.get("best_oa_location") or {}
    return {
        "title": w.get("title"),
        "year": w.get("publication_year"),
        "pdf": loc.get("pdf_url"),
        "abstract": _abstract(w.get("abstract_inverted_index")),
    }


def _abstract(inverted):
    if not inverted:
        return ""
    words = sorted((pos, word) for word, positions in inverted.items() for pos in positions)
    return " ".join(word for _, word in words)


def search_arxiv(query, n=15):
    r = requests.get(
        "http://export.arxiv.org/api/query",
        params={"search_query": f"all:{query}", "max_results": n},
        timeout=30,
    )
    r.raise_for_status()
    ns = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    out = []
    for e in ET.fromstring(r.text).findall("a:entry", ns):
        arxiv_id = e.findtext("a:id", default="", namespaces=ns).rsplit("/abs/", 1)[-1]
        doi = e.findtext("arxiv:doi", default=None, namespaces=ns)
        out.append({
            "doi": doi or f"10.48550/arXiv.{arxiv_id.split('v')[0]}",
            "title": " ".join(e.findtext("a:title", default="", namespaces=ns).split()),
            "year": int(e.findtext("a:published", default="0000", namespaces=ns)[:4]),
            "pdf": f"https://arxiv.org/pdf/{arxiv_id}",
            "abstract": " ".join(e.findtext("a:summary", default="", namespaces=ns).split()),
            "source": "arxiv",
        })
    return out


def search_family(family, per_query=10):
    """All papers for one scout family, deduplicated by DOI."""
    seen, papers = set(), []
    for q in QUERIES[family]:
        for p in search_openalex(q, per_query) + search_arxiv(q, per_query // 2):
            key = (p["doi"] or p["title"] or "").lower()
            if key and key not in seen:
                seen.add(key)
                papers.append({**p, "family": family, "query": q})
    return papers


if __name__ == "__main__":
    for p in search_family("halides", per_query=3)[:5]:
        print(p["source"], p["doi"], p["title"], bool(p["pdf"]))
