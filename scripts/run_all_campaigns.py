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
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "results" / "logs"

LANES = {
    "A": ("campaigns", [("bench", 2), ("bench", 0), ("bench", 1),
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


def wait_for(pid: int | None) -> None:
    """Block until a campaign started by an earlier launch has exited (never interrupt it)."""
    while pid:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(15)


def parse_plan(text: str) -> list[tuple[str, int]]:
    """'bench:5,ablation_nolit:1' -> [("bench", 5), ("ablation_nolit", 1)]"""
    return [(v, int(seed)) for v, seed in (item.split(":") for item in text.split(",") if item)]


def lane(name: str, mistral_keys: str, plan: list[tuple[str, int]], task: str, dry: bool,
         after_pid: int | None = None) -> None:
    wait_for(None if dry else after_pid)
    log = LOGS / f"lane-{name}.log"
    for variant, seed in plan:
        cmd = [sys.executable, "scripts/run_campaign.py", "--variant", variant, "--seed", str(seed),
               "--task", task, "--mistral-keys", mistral_keys]
        if dry:
            print(f"[dry-run] lane {name}: {' '.join(cmd[1:])}")
            continue
        for attempt in (1, 2):  # one resume after a crash (the campaign continues from its record)
            print(f"[{time.strftime('%H:%M')}] lane {name}: start {variant} s{seed}", flush=True)
            with log.open("a") as fh:
                fh.write(f"\n===== {time.strftime('%H:%M:%S')} {variant} seed {seed} =====\n")
                fh.flush()
                rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
            print(f"[{time.strftime('%H:%M')}] lane {name}: end   {variant} s{seed} (exit {rc})", flush=True)
            if rc == 0:
                break
            time.sleep(60)
        if rc != 0:
            # The same failure would hit every later campaign of this lane (a missing key, a
            # dead provider): stop the lane instead of burning through its plan.
            print(f"[{time.strftime('%H:%M')}] lane {name}: STOPPED after {variant} s{seed} failed twice "
                  f"(see {log.relative_to(ROOT)})", flush=True)
            return


def snapshot_timer(at: str, task: str, done: threading.Event) -> None:
    now = dt.datetime.now()
    target = dt.datetime.combine(now.date(), dt.time.fromisoformat(at))
    if target <= now:
        target += dt.timedelta(days=1)
    if not done.wait(timeout=(target - now).total_seconds()):
        analyze(f"snapshot-{at.replace(':', '')}", task)


# Keys the Omnigent host daemon must carry for the campaign lanes. It inherits only
# standard names and OMNIGENT_-prefixed ones (see CUSTOM_KEYS in run_campaign.py).
NEEDED_KEYS = ("MISTRAL_API_KEY", "OMNIGENT_MISTRAL_API_KEY_CAMPAIGNS", "OMNIGENT_OPENROUTER_API_KEY_CAMPAIGNS")


def stale_omnigent() -> list[str]:
    """Running Omnigent host daemons that lack the campaign keys.

    The daemon is long-lived and reused by every `omnigent run`, so one started
    earlier from another shell makes every session of a lane fail. Checks variable
    names only, never values (macOS `ps -E`)."""
    try:
        pids = subprocess.run(["pgrep", "-f", "omnigent.host._daemon_entry"],
                              capture_output=True, text=True).stdout.split()
    except OSError:
        return []
    stale = []
    for pid in pids:
        env = subprocess.run(["ps", "-E", "-ww", "-o", "command=", "-p", pid], capture_output=True, text=True).stdout
        missing = [k for k in NEEDED_KEYS if f" {k}=" not in env]
        if missing:
            stale.append(f"pid {pid} lacks {', '.join(missing)}")
    return stale


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="main")
    ap.add_argument("--snapshot-at", default="05:00", help="local time for the safety-net analysis")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--lane-a", help="override lane A's plan, e.g. bench:5,bench:6,ablation_nolit:1")
    ap.add_argument("--lane-b", help="override lane B's plan")
    ap.add_argument("--after-pid-a", type=int, help="start lane A only after this campaign process exits")
    ap.add_argument("--after-pid-b", type=int, help="start lane B only after this campaign process exits")
    args = ap.parse_args()
    lanes = {"A": (LANES["A"][0], parse_plan(args.lane_a) if args.lane_a else LANES["A"][1], args.after_pid_a),
             "B": (LANES["B"][0], parse_plan(args.lane_b) if args.lane_b else LANES["B"][1], args.after_pid_b)}
    LOGS.mkdir(parents=True, exist_ok=True)
    stale = stale_omnigent()
    if stale and not args.dry_run:
        sys.exit("The running Omnigent host daemon was started without the campaign keys "
                 f"({'; '.join(stale)}). Run `omnigent stop`, then launch again with `make campaigns`.")

    done = threading.Event()
    if not args.dry_run:
        threading.Thread(target=snapshot_timer, args=(args.snapshot_at, args.task, done), daemon=True).start()
    workers = [threading.Thread(target=lane, args=(n, keys, plan, args.task, args.dry_run, pid))
               for n, (keys, plan, pid) in lanes.items()]
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
