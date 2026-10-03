"""EDA of the OBELiX dataset.

Builds the measurement pool used by the oracle and prints the numbers needed
to fix the success threshold and k (gate f0-7).

Outputs:
  data/pool.csv              one row per material (hidden log_sigma lives here; only oracle.py may read it)
  results/eda_summary.json   threshold, k and dataset stats

Usage:
  python scripts/eda_obelix.py
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/obelix/data/downloads/all.xlsx"
BUDGET = 50
SUPERIONIC = 1e-3  # S/cm
MAJOR_FRACTION = 0.04  # elements below this atomic fraction are treated as dopants


def reduced(formula):
    try:
        return Composition(str(formula)).reduced_formula
    except Exception:
        return None


def host_system(formula):
    """Chemical system of the host lattice, ignoring minor dopants.

    Li6.5La3Zr1.5Ta0.5O12 and Li7La3Zr2O12 both map to La-Li-O-Zr.
    """
    try:
        comp = Composition(str(formula)).fractional_composition
    except Exception:
        return None
    return "-".join(sorted(el.symbol for el, x in comp.items() if x >= MAJOR_FRACTION))


def random_measurements_to_k(is_target, k, budget, seeds=2000, rng=None):
    """Measurements a random strategy needs to find k targets (inf if not within budget)."""
    rng = rng or np.random.default_rng(0)
    out = []
    for _ in range(seeds):
        order = rng.permutation(len(is_target))
        hits = np.cumsum(is_target[order])
        idx = np.argmax(hits >= k) if hits[-1] >= k else None
        n = idx + 1 if idx is not None else math.inf
        out.append(n if n <= budget else math.inf)
    return np.array(out)


def main():
    df = pd.read_excel(RAW)
    df = df.rename(columns={
        "ID": "id",
        "True Composition": "composition",
        "Reduced Composition": "reduced_obelix",
        "Ionic conductivity (S cm-1)": "sigma",
        "Family": "family",
        "Space group #": "space_group",
        "DOI": "doi",
    })
    # "<1E-10" style entries are upper bounds: keep them at the bound (never targets).
    raw_sigma = df["sigma"].astype(str).str.strip()
    df["sigma_upper_bound"] = raw_sigma.str.startswith("<")
    df["sigma"] = pd.to_numeric(raw_sigma.str.lstrip("<"), errors="coerce")
    df = df[df["sigma"] > 0].copy()
    df["log_sigma"] = np.log10(df["sigma"])
    df["reduced"] = df["composition"].apply(reduced)
    df["host"] = df["composition"].apply(host_system)
    df["family"] = df["family"].fillna("unknown").astype(str).str.strip()
    df["group_id"] = df["family"] + "|" + df["host"].fillna("?")
    df["has_cif"] = df["Cif ID"].astype(str).str.lower().eq("done")

    n = len(df)
    n_super = int((df["sigma"] >= SUPERIONIC).sum())
    frac_super = n_super / n
    top5 = float(df["log_sigma"].quantile(0.95))

    # Threshold rule from the plan: if >~10% of the pool is superionic, random finds them too fast.
    if frac_super > 0.10:
        threshold_log = top5
        threshold_rule = "top 5% of log sigma"
    else:
        threshold_log = math.log10(SUPERIONIC)
        threshold_rule = "sigma >= 1e-3 S/cm"
    df["is_target"] = df["log_sigma"] >= threshold_log

    targets = df["is_target"].to_numpy()
    n_targets = int(targets.sum())

    # Pick k so random needs a large share of the budget (median >= ~60% of budget).
    k_table = {}
    for k in range(1, min(n_targets, 15) + 1):
        runs = random_measurements_to_k(targets, k, BUDGET)
        found = np.isfinite(runs)
        k_table[k] = {
            "random_success_rate": float(found.mean()),
            "random_median": float(np.median(runs[found])) if found.any() else None,
        }
    k_choice = next(
        (k for k, v in k_table.items()
         if v["random_median"] is None or v["random_median"] >= 0.6 * BUDGET or v["random_success_rate"] < 0.8),
        max(k_table),
    )

    families = df["family"].value_counts()
    target_families = df.loc[df["is_target"], "family"].value_counts()
    groups = df["group_id"].value_counts()

    summary = {
        "n_materials": n,
        "n_with_cif": int(df["has_cif"].sum()),
        "n_superionic_1e-3": n_super,
        "frac_superionic_1e-3": round(frac_super, 4),
        "log_sigma_top5pct": round(top5, 3),
        "threshold_rule": threshold_rule,
        "threshold_log_sigma": round(threshold_log, 3),
        "n_targets": n_targets,
        "budget": BUDGET,
        "k": k_choice,
        "k_table_random": k_table,
        "n_unique_reduced_formulas": int(df["reduced"].nunique()),
        "n_near_duplicate_groups": int(groups.size),
        "n_groups_with_duplicates": int((groups > 1).sum()),
        "n_families": int(families.size),
        "families": families.to_dict(),
        "target_families": target_families.to_dict(),
        "n_distinct_target_families": int(target_families.size),
        "log_sigma_describe": df["log_sigma"].describe().round(3).to_dict(),
    }

    cols = ["id", "composition", "reduced", "host", "family", "group_id", "space_group",
            "has_cif", "doi", "log_sigma", "sigma_upper_bound", "is_target"]
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    df[cols].to_csv(ROOT / "data/pool.csv", index=False)
    (ROOT / "results/eda_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps({k: v for k, v in summary.items() if k not in ("families", "k_table_random")}, indent=2))
    print("\nRandom baseline, measurements to find k targets (budget 50):")
    for k, v in k_table.items():
        print(f"  k={k:2d}  success={v['random_success_rate']:.0%}  median={v['random_median']}")
    print("\nTop families:\n", families.head(15).to_string())


if __name__ == "__main__":
    main()
