"""Run every agent-free baseline over many seeds and save summaries + figures.

    python -m lab.run_baselines --tasks main ood classic --seeds 50 --budget 150

Baselines run past the 50-measurement agent budget so measurements-to-k is
uncensored for them; curves and hit counts are reported at the agent horizon.
Publish the curves to the dashboard with `python -m lab.publish`.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lab.baselines import STRATEGIES, run_campaign
from lab.data import ROOT, primary_k, task_pool, threshold
from lab.features import feature_table
from lab.metrics import random_expected_to_k, summarize

OUT = ROOT / "results"


def _job(args):
    name, task, seed, budget = args
    return run_campaign(STRATEGIES[name](), task, seed, budget=budget)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["main", "ood"])
    ap.add_argument("--strategies", nargs="+", default=list(STRATEGIES))
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--budget", type=int, default=150, help="run past 50 so baselines are uncensored")
    ap.add_argument("--horizon", type=int, default=50, help="agent budget used for curves")
    args = ap.parse_args()

    feature_table()  # build the cache once before forking
    OUT.mkdir(exist_ok=True)
    k = primary_k()
    for task in args.tasks:
        t0 = time.time()
        jobs = [(s, task, seed, args.budget) for s in args.strategies for seed in range(args.seeds)]
        with ProcessPoolExecutor() as ex:
            runs = list(ex.map(_job, jobs))
        pool = task_pool(task)
        n, K = len(pool), int(pool.is_hit.sum())
        summary = {
            "task": task,
            "threshold_log_sigma": threshold(task),
            "primary_k": k,
            "pool_size": n,
            "hits_in_pool": K,
            "horizon": args.horizon,
            "random_expected": {f"to_k{kk}": random_expected_to_k(n, K, kk) for kk in (3, 5, 10)},
            "strategies": {s: summarize([r for r in runs if r["strategy"] == s], horizon=args.horizon)
                           for s in args.strategies},
        }
        (OUT / f"baselines_{task}.json").write_text(json.dumps(summary, indent=1))
        (OUT / f"baselines_{task}_runs.json").write_text(json.dumps(runs))
        _plot(summary)
        _print(summary, time.time() - t0)


def _print(summary: dict, dt: float) -> None:
    k, h = summary["primary_k"], summary["horizon"]
    print(f"\n== {summary['task']}  pool={summary['pool_size']} hits={summary['hits_in_pool']}  "
          f"threshold={summary['threshold_log_sigma']:.3f}  ({dt:.0f}s)")
    exp = summary["random_expected"]
    print(f"random analytic: to_k3={exp['to_k3']:.1f} to_k5={exp['to_k5']:.1f} to_k10={exp['to_k10']:.1f}")
    print(f"{'strategy':14s} {f'to_k{k} med [IQR]':>18s} {'reach%':>7s} {'to_k5 med [IQR]':>18s} "
          f"{f'hits@{h}':>8s} {f'families@{h}':>12s}")
    fmt = lambda d: f"{d['median']:.0f} [{d['q25']:.0f}-{d['q75']:.0f}]"
    for s, v in summary["strategies"].items():
        reach = 100 * float(np.mean([x is not None and x <= h for x in v[f"to_k{k}"]["values"]]))
        print(f"{s:14s} {fmt(v[f'to_k{k}']):>18s} {reach:>6.0f}% {fmt(v['to_k5']):>18s} "
              f"{v[f'hits_at_{h}']['median']:>8.0f} {v[f'families_at_{h}']['median']:>12.0f}")


def _plot(summary: dict) -> None:
    h = summary["horizon"]
    x = np.arange(1, h + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for s, v in summary["strategies"].items():
        for ax, key in zip(axes, ("curve", "families_curve")):
            ax.plot(x, v[f"{key}_median"], label=s, lw=2)
            ax.fill_between(x, v[f"{key}_q25"], v[f"{key}_q75"], alpha=0.15)
    axes[0].set_ylabel(f"Targets found (log σ ≥ {summary['threshold_log_sigma']:.2f})")
    axes[1].set_ylabel("Distinct families among targets found")
    for ax in axes:
        ax.set_xlabel("Measurements spent")
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False)
    fig.suptitle(f"Task '{summary['task']}': {summary['hits_in_pool']} targets in {summary['pool_size']} "
                 f"candidates (median and IQR over seeds)")
    fig.tight_layout()
    fig.savefig(OUT / f"baselines_{summary['task']}.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
