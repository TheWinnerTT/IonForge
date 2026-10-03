"""Publish discovery curves to the dashboard (Supabase `curves` table).

One row per (strategy, metric, n_measured) with median and IQR over seeds, for the
baselines and for every agent campaign strategy. Previous non-demo rows of the
published strategies are replaced, so re-running is safe.

    python -m lab.publish --task main
"""
from __future__ import annotations

import argparse
import json

from integrations.supabase_sync import db
from lab.data import ROOT

OUT = ROOT / "results"


def curve_rows(strategy: str, summary: dict, n_seeds: int) -> list[dict]:
    rows = [{"strategy": strategy, "metric": "found", "n_measured": 0, "median": 0, "q25": 0, "q75": 0,
             "n_seeds": n_seeds, "is_demo": False},
            {"strategy": strategy, "metric": "families", "n_measured": 0, "median": 0, "q25": 0, "q75": 0,
             "n_seeds": n_seeds, "is_demo": False}]
    for metric, key in (("found", "curve"), ("families", "families_curve")):
        for i, (m, lo, hi) in enumerate(zip(summary[f"{key}_median"], summary[f"{key}_q25"], summary[f"{key}_q75"])):
            rows.append({"strategy": strategy, "metric": metric, "n_measured": i + 1, "median": m, "q25": lo,
                         "q75": hi, "n_seeds": n_seeds, "is_demo": False})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="main")
    args = ap.parse_args()
    if not db.enabled:
        raise SystemExit("SUPABASE_URL / SUPABASE_SERVICE_KEY not set in .env")
    rows: list[dict] = []
    base = json.loads((OUT / f"baselines_{args.task}.json").read_text())
    for s, v in base["strategies"].items():
        rows += curve_rows(s, v, v["n_seeds"])
    camp_path = OUT / f"campaigns_{args.task}.json"
    if camp_path.exists():
        for s, v in json.loads(camp_path.read_text())["strategies"].items():
            rows += curve_rows(s, v["summary"], v["summary"]["n_seeds"])
    strategies = sorted({r["strategy"] for r in rows})
    db.delete("curves", {"strategy": f"in.({','.join(strategies)})", "is_demo": "eq.false"})
    for i in range(0, len(rows), 500):
        db.insert("curves", rows[i:i + 500])
    print(f"published {len(rows)} curve rows for {', '.join(strategies)} (task {args.task})")


if __name__ == "__main__":
    main()
