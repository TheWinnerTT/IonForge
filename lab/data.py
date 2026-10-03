"""The measurement pool and task definitions.

Single source of truth: `data/pool.csv`, built by `scripts/eda_obelix.py` (gate
f0-7). One row per OBELiX entry; `candidate_id` is the OBELiX ID. The hidden
`log_sigma` column is read only through `lab.oracle.Oracle`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
POOL_CSV = ROOT / "data" / "pool.csv"
EDA_SUMMARY = ROOT / "results" / "eda_summary.json"

# Families a solid-state chemist would already bet on. Used by the out-of-distribution
# task and the expert heuristic; fixed a priori, never tuned on OBELiX.
KNOWN_SUPERIONIC_FAMILIES = ("LGPS", "argyrodites")
PUBLIC_COLUMNS = ["composition", "reduced", "family", "group_id", "space_group", "has_cif"]


@dataclass(frozen=True)
class Task:
    name: str
    threshold_log_sigma: float | None  # a hit is log10(sigma) >= threshold
    exclude_families: tuple[str, ...] = ()
    description: str = ""
    top_fraction: float | None = None  # if set, threshold = top fraction of the full pool


TASKS: dict[str, Task] = {
    # sigma >= 1e-3 S/cm puts 16.7% of the pool over the bar (docs/EDA.md), so the
    # primary target is the top 5% of log10(sigma), as in the War Room plan.
    "main": Task(
        "main",
        None,
        top_fraction=0.05,
        description="Find top-5% ionic conductors (log10 sigma >= -2.316, i.e. sigma >= 4.8e-3 S/cm).",
    ),
    "ood": Task(
        "ood",
        -3.0,
        exclude_families=KNOWN_SUPERIONIC_FAMILIES,
        description="Find superionics (sigma >= 1e-3 S/cm) outside the LGPS and argyrodite families.",
    ),
    "classic": Task(
        "classic",
        -3.0,
        description="Superionic threshold 1e-3 S/cm on the full pool (easy: 16.7% hits).",
    ),
}


@lru_cache(maxsize=1)
def load_pool() -> pd.DataFrame:
    """Public metadata + hidden target for every candidate."""
    if not POOL_CSV.exists():
        raise FileNotFoundError(f"{POOL_CSV} missing: run `python scripts/eda_obelix.py` first")
    pool = pd.read_csv(POOL_CSV).rename(columns={"id": "candidate_id"})
    pool["candidate_id"] = pool["candidate_id"].astype(str)
    pool["volume"] = _cell_volume(pool)
    return pool


@lru_cache(maxsize=1)
def primary_k() -> int:
    """k chosen in the EDA gate (smallest k where random needs >= 60% of the budget)."""
    try:
        return int(json.loads(EDA_SUMMARY.read_text())["k"])
    except (FileNotFoundError, KeyError):
        return 3


def threshold(task: Task | str) -> float:
    if isinstance(task, str):
        task = TASKS[task]
    if task.threshold_log_sigma is not None:
        return task.threshold_log_sigma
    return float(load_pool()["log_sigma"].quantile(1 - task.top_fraction))


def task_pool(task: Task | str) -> pd.DataFrame:
    if isinstance(task, str):
        task = TASKS[task]
    pool = load_pool()
    if task.exclude_families:
        pool = pool[~pool["family"].isin(task.exclude_families)]
    pool = pool.reset_index(drop=True).copy()
    pool["is_hit"] = pool["log_sigma"] >= threshold(task)
    return pool


def empirical_noise() -> float:
    """Std of log10(sigma) across repeat measurements of the same reduced formula."""
    return float(load_pool().groupby("reduced")["log_sigma"].std().dropna().mean())


def _cell_volume(df: pd.DataFrame) -> pd.Series:
    a, b, c = (pd.to_numeric(df[k], errors="coerce") for k in ("a", "b", "c"))
    al, be, ga = (np.radians(pd.to_numeric(df[k], errors="coerce")) for k in ("alpha", "beta", "gamma"))
    cos = np.cos
    term = 1 - cos(al) ** 2 - cos(be) ** 2 - cos(ga) ** 2 + 2 * cos(al) * cos(be) * cos(ga)
    return a * b * c * np.sqrt(term.clip(lower=0))


if __name__ == "__main__":
    for t in TASKS.values():
        p = task_pool(t)
        print(f"{t.name:8s} thr={threshold(t):6.3f} pool={len(p):4d} hits={int(p.is_hit.sum()):3d} "
              f"({p.is_hit.mean():.1%}) families={p[p.is_hit].family.nunique()}  {t.description}")
    print(f"primary k = {primary_k()}  |  repeat-measurement noise (std log10 sigma) = {empirical_noise():.2f}")
