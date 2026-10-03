"""Run a discovery campaign as one Omnigent session per round.

Why per round: a headless `omnigent run -p` follows an async orchestrator for at most
30 minutes / 30 wake-ups, while a 10-round campaign takes ~45 minutes. A fresh session
per round also keeps the PI's context from growing round after round (the main cost in
the first test). The lab state lives in the research record (results/runs/<run_id>.jsonl),
so each round starts from where the previous one ended.

    python scripts/run_campaign.py --variant bench --seed 0          # full campaign (50 measurements)
    python scripts/run_campaign.py --variant live --max-rounds 3     # live demo, 3 rounds
    python scripts/run_campaign.py --variant smoke                   # 2-round end-to-end test
    python scripts/run_campaign.py --variant bench --seed 0 --dry-run

Stops when the budget is spent, after --max-rounds, or after two consecutive rounds that
measured nothing (so a broken setup cannot keep spending credits). Every round's output
is kept in results/logs/<run_id>/round-<n>.log.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")  # values already exported by `make` take precedence

from lab.oracle import Oracle  # noqa: E402

OMNI = os.path.expanduser("~/.local/bin/omnigent")
ROUND_TIMEOUT_S = 25 * 60
MAX_FAILED_ROUNDS = 2


def build(variant: str, task: str, seed: int, budget: int, usd_cap: float, run_id: str | None = None) -> Path:
    if run_id and (ROOT / "agents" / "build" / run_id / "config.yaml").exists():
        return ROOT / "agents" / "build" / run_id  # continue an existing campaign: same bundle, same record
    extra = ["--run-id", run_id] if run_id else []  # e.g. a fresh CI machine: rebuild under the same name
    out = subprocess.run(
        [sys.executable, str(ROOT / "agents" / "build.py"), "--variant", variant, "--task", task,
         "--seed", str(seed), "--budget", str(budget), "--usd-cap", str(usd_cap), *extra],
        capture_output=True, text=True, cwd=ROOT,
    )
    if out.returncode != 0:
        sys.exit(f"build failed: {out.stderr.strip() or out.stdout.strip()}")
    return Path(out.stdout.strip().splitlines()[-1])


def lab_state(task: str, budget: int, seed: int, run_id: str, strategy: str) -> dict:
    """Read-only view of the research record (no API call)."""
    path = ROOT / "results" / "runs" / f"{run_id}.jsonl"
    if not path.exists():
        return {"measured": 0, "remaining": budget, "round": 0, "hits": 0, "families": 0, "closed": False,
                "critic_reviewed_last_round": False}
    lab = Oracle(task, budget=budget, seed=seed, run_id=run_id, strategy=strategy)
    closed = any(e.get("event") == "next_step" for e in lab.events(("next_step",)))
    reviewed = {e.get("round") for e in lab.events(("review",)) if e.get("author") == "critic"}
    return {"measured": len(lab.measured), "remaining": lab.remaining, "round": lab.round,
            "hits": lab.n_hits(), "families": len(lab.families_found()), "closed": closed,
            "critic_reviewed_last_round": lab.round in reviewed}


def round_prompt(n: int, state: dict) -> str:
    if state["remaining"] < 1:
        return ("The measurement budget is spent. Record the final next_step for this campaign "
                "(best candidates, what was learned, what remains uncertain, next experiment) and stop.")
    return (f"Run round {n} of the discovery campaign ({state['remaining']:.0f} measurements left, "
            f"{state['measured']} measured so far). Follow your instructions for exactly one round, then stop.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="bench",
                    choices=["live", "live_judges", "smoke", "bench", "ablation_nolit", "ablation_anon"])
    ap.add_argument("--task", default="main")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--max-rounds", type=int, default=0, help="stop after this many rounds (0 = until budget)")
    ap.add_argument("--usd-cap", type=float, default=1.5, help="hard spend cap per round session (USD)")
    ap.add_argument("--dry-run", action="store_true", help="render and print the plan; call nothing")
    ap.add_argument("--run-id", help="continue this campaign (reuses agents/build/<run_id>/ and its record)")
    args = ap.parse_args()

    if args.variant == "smoke":
        args.budget = min(args.budget, 10)
    bundle = build(args.variant, args.task, args.seed, args.budget, args.usd_cap, args.run_id)
    run_id = bundle.name
    strategy = {"ablation_nolit": "ablation_no_lit", "ablation_anon": "ablation_anon"}.get(args.variant, "ionforge")
    logs = ROOT / "results" / "logs" / run_id
    logs.mkdir(parents=True, exist_ok=True)
    max_rounds = args.max_rounds or (args.budget // 5 + 1)  # +1: the closing next_step session

    print(f"campaign {run_id}: variant={args.variant} budget={args.budget} max_rounds={max_rounds} "
          f"cap=${args.usd_cap:.2f}/round", flush=True)
    failed = 0
    for n in range(1, max_rounds + 1):
        state = lab_state(args.task, args.budget, args.seed, run_id, strategy)
        if state["remaining"] < 1 and state["closed"]:
            print("budget spent and campaign closed", flush=True)
            break
        prompt = round_prompt(state["round"] + 1, state)
        cmd = [OMNI, "run", str(bundle), "-p", prompt]
        if args.dry_run:
            print(f"[dry-run] round {n}: {' '.join(cmd[:3])} -p {prompt!r}")
            if n >= 2:
                break
            continue
        t0 = time.time()
        # numbered by the lab's round, not this loop's counter: continued campaigns keep every log
        log = logs / f"round-{state['round'] + 1:02d}.log"
        if log.exists():
            log = logs / f"round-{state['round'] + 1:02d}-{time.strftime('%H%M%S')}.log"
        with log.open("w") as fh:
            try:
                rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT,
                                    env=os.environ.copy(), timeout=ROUND_TIMEOUT_S).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        after = lab_state(args.task, args.budget, args.seed, run_id, strategy)
        progressed = after["measured"] > state["measured"] or (state["remaining"] < 1 and after["closed"])
        print(json.dumps({"round": n, "rc": rc, "minutes": round((time.time() - t0) / 60, 1),
                          "measured": after["measured"], "hits": after["hits"],
                          "families": after["families"], "remaining": after["remaining"],
                          "critic_review": after["critic_reviewed_last_round"],
                          "log": str(log.relative_to(ROOT))}), flush=True)
        if after["measured"] > state["measured"] and not after["critic_reviewed_last_round"]:
            print(f"WARNING: round {after['round']} measured without a critic review", flush=True)
        failed = 0 if progressed else failed + 1
        if failed >= MAX_FAILED_ROUNDS:
            sys.exit(f"stopping: {failed} consecutive rounds without progress (see {logs})")
        if after["remaining"] < 1 and after["closed"]:
            print("budget spent and campaign closed", flush=True)
            break


if __name__ == "__main__":
    main()
