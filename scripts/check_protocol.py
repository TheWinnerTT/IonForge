"""Protocol check for agent campaigns, decided before looking at any result.

A campaign is valid when:
  1. round 1 has literature evidence (except the no-literature ablation);
  2. every round r has, recorded for round r: hypotheses, candidate sets, the planner's
     experiment spec, the safety review, the measurement and the Critic's review;
  3. every round except the last has the PI director's direction for round r+1;
  4. the whole budget is measured;
  5. the campaign ends with its closing next_step.

Results (hits, conductivities) are never read.

    python scripts/check_protocol.py                       # every campaign in results/runs/
    python scripts/check_protocol.py ionforge-main-s2      # one campaign
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
STAGES = [("hypothesis", None, "hypotheses"), ("candidate_set", None, "candidate sets"),
          ("experiment_spec", None, "experiment spec"), ("review", "safety_officer", "safety review"),
          ("measure", None, "measurement"), ("review", "critic", "Critic review")]


def check(path: Path) -> list[str]:
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    start = next((e for e in events if e.get("event") == "run_start"), {})
    budget = start.get("budget", 50)
    strategy = start.get("strategy", "")

    def has(kind: str, rnd: int, author: str | None = None) -> bool:
        return any(e.get("event") == kind and e.get("round") == rnd
                   and (author is None or e.get("author") == author or e.get("requested_by") == author)
                   for e in events)

    problems = []
    measures = [e for e in events if e.get("event") == "measure"]
    rounds = sorted({e["round"] for e in measures})
    if strategy != "ablation_no_lit" and not any(e.get("event") == "evidence" for e in events):
        problems.append("no literature evidence")
    for r in rounds:
        for kind, author, label in STAGES:
            if kind == "review" and author == "safety_officer":
                ok = any(e.get("event") == "review" and e.get("round") == r and e.get("author") != "critic"
                         for e in events)
            else:
                ok = has(kind, r, author)
            if not ok:
                problems.append(f"round {r}: no {label}")
        if r != rounds[-1] and not has("direction", r + 1):
            problems.append(f"round {r}: no PI direction for round {r + 1}")
    measured = sum(len(e.get("results") or []) for e in measures)
    if measured < budget:
        problems.append(f"measured {measured} of {budget}")
    if not any(e.get("event") == "next_step" for e in events):
        problems.append("no closing next_step")
    return problems


def main() -> None:
    names = sys.argv[1:] or sorted(p.stem for p in RUNS.glob("*.jsonl")
                                   if "-smoke-" not in p.stem and "-live-" not in p.stem)
    bad = 0
    for name in names:
        problems = check(RUNS / f"{name}.jsonl")
        bad += bool(problems)
        print(f"{'VALID  ' if not problems else 'INVALID'} {name}" + "".join(f"\n          - {p}" for p in problems))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
