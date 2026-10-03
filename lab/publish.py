"""Publish discovery curves to the dashboard (Supabase `curves` table).

One row per (strategy, metric, n_measured) with median and IQR over seeds, for the
baselines and for every agent campaign strategy. Previous non-demo rows of the
published strategies are replaced, so re-running is safe.

Refuses to publish while the fake rows from scripts/seed_demo_data.py are still in
Supabase: the fake `ionforge` curve has an invented speed-up and must never be shown
next to measured ones. Clear them with `python scripts/clear_demo_data.py`.

    python -m lab.publish --task main
    python -m lab.publish --check-demo     # only report whether fake rows remain
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


DEMO_TABLES = ("runs", "curves", "evidence", "candidates")


def demo_rows_present() -> list[str]:
    """Tables that still hold rows seeded by scripts/seed_demo_data.py."""
    return [t for t in DEMO_TABLES if db.select(t, {"is_demo": "true"}, columns="is_demo", limit=1)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="main")
    ap.add_argument("--check-demo", action="store_true", help="only report fake demo rows")
    args = ap.parse_args()
    if not db.enabled:
        raise SystemExit("SUPABASE_URL / SUPABASE_SERVICE_KEY not set in .env")
    demo = demo_rows_present()
    if args.check_demo:
        print(f"WARNING: fake demo rows still in Supabase ({', '.join(demo)}): run "
              f"`python scripts/clear_demo_data.py` before the real demo" if demo else "no fake demo rows")
        return
    if demo:
        raise SystemExit(f"Refusing to publish: fake demo rows are still in Supabase ({', '.join(demo)}). "
                         f"Run `python scripts/clear_demo_data.py` first.")
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
