"""Which descriptor set should the surrogate use? Decide with data.

1. Repeated 5-fold cross-validation of the random-forest surrogate on log10(sigma):
   RMSE, Spearman rank correlation, and recall of the top-5% conductors among the
   model's own top-5% predictions.
2. Downstream: BO + prior knowledge (the fair baseline) run with each set.

    python -m lab.compare_features --seeds 20
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from lab.baselines import BOStrategy, run_campaign
from lab.data import ROOT, load_pool, threshold
from lab.features import feature_table
from lab.metrics import summarize

SETS = ("basic", "matminer", "matminer_chem")


def cross_validate(feature_set: str, repeats: int = 5) -> dict:
    pool = load_pool().set_index("candidate_id")
    X = feature_table(feature_set).loc[pool.index].to_numpy(float)
    y = pool["log_sigma"].to_numpy(float)
    top = y >= np.quantile(y, 0.95)
    rmse, rho, recall = [], [], []
    for r in range(repeats):
        pred = np.zeros_like(y)
        for tr, te in KFold(5, shuffle=True, random_state=r).split(X):
            m = RandomForestRegressor(300, min_samples_leaf=2, max_features=0.5, random_state=r, n_jobs=-1)
            pred[te] = m.fit(X[tr], y[tr]).predict(X[te])
        rmse.append(float(np.sqrt(np.mean((pred - y) ** 2))))
        rho.append(float(spearmanr(pred, y).statistic))
        k = int(top.sum())
        recall.append(float(top[np.argsort(-pred)[:k]].mean()))
    return {"n_features": X.shape[1], "rmse": _ms(rmse), "spearman": _ms(rho), "top5_recall": _ms(recall)}


def _ms(v):
    return {"mean": round(float(np.mean(v)), 3), "std": round(float(np.std(v)), 3)}


def _bo_job(args):
    feature_set, seed = args
    import lab.surrogate as s
    s.DEFAULT_FEATURE_SET = feature_set
    return run_campaign(BOStrategy(warm_start="expert"), "main", seed, budget=50)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()
    for s in SETS:
        feature_table(s)
    out = {"task": "main", "threshold_log_sigma": threshold("main"), "sets": {}}
    for s in SETS:
        cv = cross_validate(s)
        with ProcessPoolExecutor() as ex:
            runs = list(ex.map(_bo_job, [(s, seed) for seed in range(args.seeds)]))
        bo = summarize(runs, horizon=50)
        out["sets"][s] = {**cv, "bo_prior": {k: bo[k] for k in ("to_k3", "to_k5", "hits_at_50", "families_at_50")}}
        b = out["sets"][s]["bo_prior"]
        print(f"{s:14s} n={cv['n_features']:3d}  RMSE={cv['rmse']['mean']:.2f}±{cv['rmse']['std']:.2f}  "
              f"Spearman={cv['spearman']['mean']:.3f}  top5%-recall={cv['top5_recall']['mean']:.2f}  | "
              f"BO+prior to_k3={b['to_k3']['median']:.0f} to_k5={b['to_k5']['median']:.0f} "
              f"hits@50={b['hits_at_50']['median']:.0f} families@50={b['families_at_50']['median']:.0f}")
    (ROOT / "results" / "feature_comparison.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
