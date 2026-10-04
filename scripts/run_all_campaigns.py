"""Overnight plan: every campaign, in two parallel lanes, IonForge seeds first.

Each lane runs its campaigns one after another on its own Mistral key (separate rate
limits); the Critic always uses the campaign OpenRouter key, so the demo budget is never
touched. Campaigns resume from their research record, so re-running this script after
a crash continues where it stopped (finished campaigns are skipped in seconds).

  lane A (Mistral campaign key):  IonForge s0, s1, s2  -> Ablation 1 s0, s1, s2
  lane B (Mistral demo key):      IonForge s3, s4      -> Ablation 2 s0, s1, s2

A snapshot `lab.analyze` runs at --snapshot-at (default 05:00) and again when both lanes
finish, so there is always a number to show even if something fails late.

    python scripts/run_all_campaigns.py
    python scripts/run_all_campaigns.py --dry-run
"""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "results" / "logs"

LANES = {
    "A": ("campaigns", [("bench", 0), ("bench", 1), ("bench", 2),
                        ("ablation_nolit", 0), ("ablation_nolit", 1), ("ablation_nolit", 2)]),
    "B": ("demo", [("bench", 3), ("bench", 4),
                   ("ablation_anon", 0), ("ablation_anon", 1), ("ablation_anon", 2)]),
}


def analyze(tag: str, task: str) -> None:
    out = LOGS / f"analyze-{tag}.txt"
    with out.open("w") as fh:
        subprocess.run([sys.executable, "-m", "lab.analyze", "--task", task], cwd=ROOT, stdout=fh,
                       stderr=subprocess.STDOUT)
    print(f"[{time.strftime('%H:%M')}] analyze ({tag}) -> {out.relative_to(ROOT)}", flush=True)


def lane(name: str, mistral_keys: str, plan: list[tuple[str, int]], task: str, dry: bool) -> None:
    log = LOGS / f"lane-{name}.log"
    for variant, seed in plan:
        cmd = [sys.executable, "scripts/run_campaign.py", "--variant", variant, "--seed", str(seed),
               "--task", task, "--mistral-keys", mistral_keys]
        if dry:
            print(f"[dry-run] lane {name}: {' '.join(cmd[1:])}")
            continue
        print(f"[{time.strftime('%H:%M')}] lane {name}: start {variant} s{seed}", flush=True)
        with log.open("a") as fh:
            fh.write(f"\n===== {time.strftime('%H:%M:%S')} {variant} seed {seed} =====\n")
            fh.flush()
            rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
        print(f"[{time.strftime('%H:%M')}] lane {name}: end   {variant} s{seed} (exit {rc})", flush=True)


def snapshot_timer(at: str, task: str, done: threading.Event) -> None:
    now = dt.datetime.now()
    target = dt.datetime.combine(now.date(), dt.time.fromisoformat(at))
    if target <= now:
        target += dt.timedelta(days=1)
    if not done.wait(timeout=(target - now).total_seconds()):
        analyze(f"snapshot-{at.replace(':', '')}", task)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="main")
    ap.add_argument("--snapshot-at", default="05:00", help="local time for the safety-net analysis")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)

    done = threading.Event()
    if not args.dry_run:
        threading.Thread(target=snapshot_timer, args=(args.snapshot_at, args.task, done), daemon=True).start()
    workers = [threading.Thread(target=lane, args=(n, keys, plan, args.task, args.dry_run))
               for n, (keys, plan) in LANES.items()]
    for w in workers:
        w.start()
        time.sleep(0 if args.dry_run else 20)  # stagger the two lanes' first sessions
    for w in workers:
        w.join()
    done.set()
    if not args.dry_run:
        analyze("final", args.task)


if __name__ == "__main__":
    main()
