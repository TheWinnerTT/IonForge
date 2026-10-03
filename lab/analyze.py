"""Compare the agent campaigns against the agent-free baselines.

Reads every research record in results/runs/*.jsonl, groups runs by strategy
(ionforge, ablation_no_lit, ablation_anon), and compares them with
results/baselines_<task>.json on the same task:
  * measurements to the k-th target (primary k from the EDA gate) and speed-up
    = median(baseline) / median(strategy), with a bootstrap 90% interval;
  * targets found and distinct families found within the 50-measurement budget.

Live demo and smoke-test runs (run ids containing "-live-" / "-smoke-") are excluded.

    python -m lab.analyze --task main
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lab.data import ROOT, primary_k
from lab.metrics import measurements_to_k, speedup, summarize

RUNS = ROOT / "results" / "runs"
OUT = ROOT / "results"


def load_campaigns(task: str, include_live: bool = False) -> dict[str, list[dict]]:
    by_strategy: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(RUNS.glob("*.jsonl")):
        events = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        start = next((e for e in events if e.get("event") == "run_start"), None)
        if not start or start.get("task") != task:
            continue
        if ("-live-" in start["run_id"] or "-smoke-" in start["run_id"]) and not include_live:
            continue
        results = [r for e in events if e.get("event") == "measure" for r in e["results"]]
        by_strategy[start.get("strategy", "ionforge")].append({
            "run_id": start["run_id"],
            "seed": start.get("seed"),
            "budget": start["budget"],
            "measured": len(results),
            "complete": len(results) >= start["budget"],
            "hits": [bool(r["is_hit"]) for r in results],
            "hit_families": [r["family"] if r["is_hit"] else None for r in results],
            "designs": [e.get("design") for e in events if e.get("event") == "measure"],
        })
    return dict(by_strategy)


def compare(task: str) -> dict:
    base = json.loads((OUT / f"baselines_{task}.json").read_text())
    k, h = primary_k(), base["horizon"]
    campaigns = load_campaigns(task)
    out = {"task": task, "primary_k": k, "horizon": h, "strategies": {}}
    for strat, runs in campaigns.items():
        summ = summarize(runs, horizon=h)
        mine = [measurements_to_k(r["hits"], k) for r in runs]
        vs = {}
        for b, bs in base["strategies"].items():
            bvals = [np.inf if v is None else v for v in bs[f"to_k{k}"]["values"]]
            vs[b] = {
                f"speedup_to_k{k}": speedup(bvals, mine),
                "hits_diff": summ[f"hits_at_{h}"]["median"] - bs[f"hits_at_{h}"]["median"],
                "families_diff": summ[f"families_at_{h}"]["median"] - bs[f"families_at_{h}"]["median"],
            }
        out["strategies"][strat] = {
            "runs": [{k2: r[k2] for k2 in ("run_id", "seed", "measured", "complete", "designs")} for r in runs],
            "n_complete": sum(r["complete"] for r in runs),
            "summary": summ,
            "vs_baselines": vs,
        }
    (OUT / f"campaigns_{task}.json").write_text(json.dumps(out, indent=1, default=float))
    _plot(base, out)
    return out


def _plot(base: dict, camp: dict) -> None:
    h = base["horizon"]
    x = np.arange(1, h + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    series = {**{s: v for s, v in base["strategies"].items()},
              **{s: v["summary"] for s, v in camp["strategies"].items()}}
    for s, v in series.items():
        agent = s in camp["strategies"]
        for ax, key in zip(axes, ("curve", "families_curve")):
            ax.plot(x, v[f"{key}_median"], label=s, lw=3 if agent else 1.5, ls="-" if agent else "--")
            ax.fill_between(x, v[f"{key}_q25"], v[f"{key}_q75"], alpha=0.12)
    axes[0].set_ylabel(f"Targets found (log σ ≥ {base['threshold_log_sigma']:.2f})")
    axes[1].set_ylabel("Distinct families among targets found")
    for ax in axes:
        ax.set_xlabel("Measurements spent")
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Task '{base['task']}': agent campaigns (solid) vs baselines (dashed), median and IQR")
    fig.tight_layout()
    fig.savefig(OUT / f"campaigns_{base['task']}.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="main")
    args = ap.parse_args()
    out = compare(args.task)
    k, h = out["primary_k"], out["horizon"]
    if not out["strategies"]:
        print(f"no agent campaigns for task {args.task!r} in {RUNS}")
        return
    for s, v in out["strategies"].items():
        sm = v["summary"]
        print(f"\n== {s}: {len(v['runs'])} runs ({v['n_complete']} complete)  "
              f"to_k{k} median={sm[f'to_k{k}']['median']:.0f}  hits@{h}={sm[f'hits_at_{h}']['median']:.0f}  "
              f"families@{h}={sm[f'families_at_{h}']['median']:.0f}")
        for b, d in v["vs_baselines"].items():
            sp = d[f"speedup_to_k{k}"]
            print(f"   vs {b:14s} speed-up to k={k}: {sp['point']:.2f}x [90% CI {sp['ci90'][0]:.2f}-{sp['ci90'][1]:.2f}]  "
                  f"hits {d['hits_diff']:+.0f}  families {d['families_diff']:+.0f}")


if __name__ == "__main__":
    main()
