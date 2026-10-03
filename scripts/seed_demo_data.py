"""Seed FAKE data so the Lovable dashboard has something to render.

Every row is tagged is_demo=true (directly or via a demo run) and is removed by
scripts/clear_demo_data.py. Delete it before the real demo.

Usage:
  python scripts/seed_demo_data.py            # seed
  python scripts/seed_demo_data.py --live     # also stream events slowly to test realtime
"""
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrations.supabase_sync import db  # noqa: E402

RUN = "demo-ionforge-s0"
BUDGET, ROUNDS, BATCH = 50, 10, 5
STRATEGIES = {  # fake speed factor vs random, only for the shape of the demo curves
    "random": 1.0, "heuristic": 1.6, "bo_cold": 2.0, "bo_prior": 2.6, "ionforge": 3.6,
}
AGENTS = ["literature_scout_sulfides", "literature_scout_oxides", "literature_scout_halides",
          "hypothesis_generator", "screening", "experiment_planner", "safety", "lab_runner", "critic"]


def curves():
    rows, base = [], 30 / 599  # targets per random measurement
    for strat, f in STRATEGIES.items():
        for n in range(0, BUDGET + 1):
            med = min(30, base * f * n * (1 + 0.15 * np.log1p(n) * (f > 1)))
            spread = 0.6 + 0.08 * n ** 0.5
            for metric, scale in (("found", 1.0), ("families", 0.35)):
                m = med * scale
                rows.append({"strategy": strat, "metric": metric, "n_measured": n, "median": round(m, 2),
                             "q25": round(max(0, m - spread * scale), 2), "q75": round(m + spread * scale, 2),
                             "n_seeds": 5 if strat == "ionforge" else 50, "is_demo": True})
    return rows


def main(live):
    pool = pd.read_csv(Path(__file__).resolve().parents[1] / "data/pool.csv")
    rng = random.Random(0)

    db.upsert("runs", [{"id": f"demo-{s}-s0", "strategy": s, "seed": 0, "budget": BUDGET,
                        "status": "done", "is_demo": True} for s in STRATEGIES if s != "ionforge"]
              + [{"id": RUN, "strategy": "ionforge", "seed": 0, "budget": BUDGET, "status": "running", "is_demo": True}])
    db.insert("curves", curves())

    db.upsert("evidence", [
        {"id": "demo-ev1", "doi": "10.0000/demo.1", "title": "DEMO: Halogen disorder in argyrodites", "family": "sulfides",
         "trend": "In argyrodites, more site disorder between S and Cl tends to raise Li conductivity.",
         "quote": "anion site disorder opens additional Li diffusion pathways", "verified": True, "source": "openalex", "is_demo": True},
        {"id": "demo-ev2", "doi": "10.0000/demo.2", "title": "DEMO: Bottleneck size in NASICON", "family": "oxides",
         "trend": "In NASICON frameworks, larger bottleneck windows tend to lower migration barriers.",
         "quote": "the bottleneck size correlates with the activation energy", "verified": True, "source": "arxiv", "is_demo": True},
        {"id": "demo-ev3", "doi": "10.0000/demo.3", "title": "DEMO: Paper reporting a pool value", "family": "sulfides",
         "trend": "Li6PS5Cl reaches 1.9 mS/cm.", "quote": "Li6PS5Cl shows 1.9 mS cm-1", "verified": True,
         "blocked_leak": True, "blocked_reason": "conductivity value for Li6PS5Cl (pool material)", "source": "openalex", "is_demo": True},
    ])

    db.upsert("hypotheses", [
        {"id": "demo-h1", "run_id": RUN, "round": 1, "statement": "Garnet-type oxides dominate the high-conductivity region",
         "predicted_effect": "log σ > -3 for cubic garnets", "confidence": 0.62, "status": "reopened", "evidence_ids": ["demo-ev2"]},
        {"id": "demo-h2", "run_id": RUN, "round": 3, "statement": "Halogen-rich argyrodites outperform oxides at room temperature",
         "predicted_effect": "log σ > -2.5 for Cl/Br-rich argyrodites", "confidence": 0.71, "status": "supported",
         "evidence_ids": ["demo-ev1"], "parent_id": "demo-h1"},
        {"id": "demo-h3", "run_id": RUN, "round": 4, "statement": "LGPS-type frameworks with Si/Sn substitution keep 1D channels open",
         "predicted_effect": "top 5% of log σ", "confidence": 0.55, "status": "open", "evidence_ids": []},
    ])

    db.upsert("candidates", [
        {"id": f"demo-c{i}", "formula": f, "mp_id": f"mp-demo{i}", "e_hull": eh, "band_gap": bg, "pred_log_sigma": p,
         "uncertainty": u, "in_domain": dom, "domain_distance": dd, "rank": i, "is_demo": True,
         "rationale": "DEMO candidate", "next_experiment": "AIMD at 600-1000 K, then solid-state synthesis"}
        for i, (f, eh, bg, p, u, dom, dd) in enumerate([
            ("Li3InCl6", 0.0, 4.9, -2.9, 0.8, True, 0.6), ("Li2ZrCl6", 0.02, 4.4, -3.4, 0.9, True, 0.8),
            ("Li5PS4Br2", 0.04, 2.6, -2.7, 1.1, True, 1.0), ("LiTaOCl4", 0.03, 3.9, -3.1, 1.6, False, 2.4),
        ], start=1)
    ])

    measured, found, families = 0, 0, set()
    for rnd in range(1, 5):
        design = ["exploit", "explore", "test_hypothesis", "explore"][rnd - 1]
        steps = [
            ("literature_scout_sulfides", "evidence", "3 cards on argyrodites (1 blocked by no-leak rule)"),
            ("literature_scout_oxides", "evidence", "4 cards on garnets and NASICON"),
            ("literature_scout_halides", "evidence", "2 cards on Li3MCl6 halides"),
            ("hypothesis_generator", "hypothesis", "Proposed 'halogen-rich argyrodites outperform oxides'"),
            ("screening", "candidates", "41 candidates match active hypotheses"),
            ("experiment_planner", "plan", f"Chose '{design}' design"),
            ("safety", "approval", "5 measurements within the spend cap; approval requested"),
        ]
        for agent, kind, summary in (steps if rnd == 1 else steps[3:]):
            payload = {}
            if agent == "experiment_planner":
                payload = {"designs": [{"name": d, "score": round(rng.uniform(0.2, 0.9), 2)} for d in
                                       ("exploit", "explore", "test_hypothesis")], "chosen": design}
            db.event(RUN, rnd, agent, kind, summary, payload)
            if live:
                time.sleep(1.5)
        db.insert("approvals", {"run_id": RUN, "round": rnd, "design": design,
                                "request": f"IonForge wants to spend 5 measurements on the {design} design",
                                "status": "approved", "channel": "whatsapp", "decided_by": "+56900000000"})
        batch = pool.sample(BATCH, random_state=rnd)
        rows = []
        for _, m in batch.iterrows():
            measured += 1
            if m["is_target"]:
                found += 1
                families.add(m["family"])
            rows.append({"run_id": RUN, "round": rnd, "material": m["composition"], "family": m["family"],
                         "log_sigma": round(m["log_sigma"], 2), "superionic": bool(m["is_target"]),
                         "requested_by": "experiment_planner", "reason": f"DEMO {design}",
                         "cumulative_found": found, "families_found": len(families), "n_measured": measured})
        db.insert("measurements", rows)
        db.event(RUN, rnd, "lab_runner", "measurement", f"Measured {BATCH} materials; {found} superionic so far")
        if rnd == 3:
            db.event(RUN, rnd, "critic", "reopen", "Garnet hypothesis contradicted: 0/5 garnets above threshold",
                     {"verdict": "reopen", "hypothesis_ids": ["demo-h1"]})
        else:
            db.event(RUN, rnd, "critic", "review", "Conclusions follow from measurements", {"verdict": "accept"})
    db.insert("approvals", {"run_id": RUN, "round": 5, "design": "exploit",
                            "request": "IonForge wants to spend 5 measurements on the exploit design (DEMO pending)"})
    print("Demo data seeded." if db.enabled else "Supabase not configured: printed rows only.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    main(ap.parse_args().live)
