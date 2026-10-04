"""Out-of-dataset discovery: new Li conductors from Materials Project, ranked with uncertainty.

The campaigns re-discover materials whose conductivity was already measured. This step
points the lab at materials nobody has measured as electrolytes:

  1. Materials Project: Li compounds, thermodynamically stable or nearly so
     (E_above_hull <= 0.05 eV/atom), electronic insulators (band gap >= 3 eV), 3-6 elements.
  2. Drop every reduced formula already in OBELiX.
  3. Same descriptors as the lab (matminer + Li/anion/cell chemistry), random forest trained
     on all 599 measured OBELiX materials: predicted log10 sigma, uncertainty (spread across
     trees) and probability of reaching the top-5% target.
  4. Applicability domain: mean distance to the 5 nearest OBELiX materials in standardized
     descriptor space, relative to the 95th percentile of that distance within OBELiX.
     Candidates beyond it are flagged out of domain and ranked separately.
  5. Top 10 in-domain (+ top 5 out-of-domain) -> results/candidates.json and Supabase
     `candidates`; the next experiment for the top 3 (AIMD screening, then synthesis + EIS)
     is filed as a pending human approval.

Everything here is an AI-generated hypothesis, not lab-validated.

    python -m lab.discovery                # query MP (free API, MP_API_KEY), rank, publish
    python -m lab.discovery --no-publish   # rank only, write results/ files
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from lab.data import ROOT, load_pool, threshold
from lab.features import CACHE, DEFAULT_SET, chemistry, feature_table, matminer_features

OUT = ROOT / "results"
MP_CACHE = CACHE / "mp_candidates.csv"
CRITERIA = {"elements": ["Li"], "energy_above_hull": (0.0, 0.05), "band_gap": (3.0, None),
            "num_elements": (3, 6), "num_sites": (1, 120)}
NEXT_EXPERIMENT = ("AIMD at 600-1000 K (~100 ps per temperature) to estimate Li diffusivity and "
                   "activation energy; if Ea < 0.35 eV, solid-state synthesis and room-temperature "
                   "impedance spectroscopy (EIS).")
DISCOVERY_RUN = "discovery-materials-project"


def fetch_materials_project(refresh: bool = False) -> pd.DataFrame:
    if MP_CACHE.exists() and not refresh:
        return pd.read_csv(MP_CACHE)
    from mp_api.client import MPRester

    with MPRester(os.environ["MP_API_KEY"], mute_progress_bars=True) as mpr:
        docs = mpr.materials.summary.search(
            **CRITERIA, is_metal=False,
            fields=["material_id", "formula_pretty", "energy_above_hull", "band_gap", "nsites",
                    "volume", "symmetry", "theoretical"])
    rows = [{"mp_id": str(d.material_id), "formula": d.formula_pretty, "e_hull": d.energy_above_hull,
             "band_gap": d.band_gap, "nsites": d.nsites, "volume": d.volume,
             "space_group": d.symmetry.number if d.symmetry else None, "theoretical": d.theoretical}
            for d in docs]
    df = pd.DataFrame(rows)
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_csv(MP_CACHE, index=False)
    return df


def novel(mp: pd.DataFrame) -> pd.DataFrame:
    """Keep formulas absent from OBELiX; one entry per formula (lowest E_hull)."""
    from pymatgen.core import Composition

    known = set(load_pool()["reduced"].dropna())
    mp = mp.copy()
    mp["reduced"] = [Composition(f).reduced_formula for f in mp["formula"]]
    mp = mp[~mp["reduced"].isin(known)]
    return mp.sort_values("e_hull").drop_duplicates("reduced").reset_index(drop=True)


def descriptors(mp: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """The lab's descriptor set (DEFAULT_SET) for Materials Project entries."""
    formulas = list(mp["formula"])
    chem = pd.DataFrame([chemistry(f) for f in formulas], index=formulas)
    mm = matminer_features(formulas, n_jobs=1)  # no process pool: safe from any caller
    X = pd.concat([mm, chem], axis=1)
    X.index = mp.index
    X["space_group"] = mp["space_group"].astype(float)
    X["vol_per_atom"] = mp["volume"] / mp["nsites"]
    X["li_density"] = X["li_frac"] / X["vol_per_atom"]
    return X.reindex(columns=columns)


def rank(mp: pd.DataFrame, n_trees: int = 400, seed: int = 0) -> pd.DataFrame:
    pool = load_pool()
    train = feature_table(DEFAULT_SET).loc[pool["candidate_id"]]
    y = pool["log_sigma"].to_numpy(float)
    X_new = descriptors(mp, list(train.columns)).fillna(train.median(numeric_only=True))

    rf = RandomForestRegressor(n_trees, min_samples_leaf=2, max_features=0.5, random_state=seed, n_jobs=-1)
    rf.fit(train.to_numpy(float), y)
    per_tree = np.stack([t.predict(X_new.to_numpy(float)) for t in rf.estimators_])
    mu, sd = per_tree.mean(0), per_tree.std(0) + 1e-6
    thr = threshold("main")

    scaler = StandardScaler().fit(train.to_numpy(float))
    Z_train, Z_new = scaler.transform(train.to_numpy(float)), scaler.transform(X_new.to_numpy(float))
    nn = NearestNeighbors(n_neighbors=6).fit(Z_train)
    d_train = nn.kneighbors(Z_train)[0][:, 1:].mean(1)          # leave-self-out
    d_new, idx_new = nn.kneighbors(Z_new, n_neighbors=5)
    cutoff = float(np.quantile(d_train, 0.95))

    out = mp.copy()
    out["pred_log_sigma"], out["uncertainty"] = mu, sd
    out["p_target"] = 1 - norm.cdf((thr - mu) / sd)
    out["domain_distance"] = d_new.mean(1) / cutoff              # <= 1: within OBELiX's own spread
    out["in_domain"] = out["domain_distance"] <= 1.0
    nearest = pool.iloc[idx_new[:, 0]].reset_index(drop=True)
    out["nearest_obelix"] = nearest["composition"].values
    out["nearest_family"] = nearest["family"].values
    out["nearest_log_sigma"] = nearest["log_sigma"].round(2).values
    out["li_frac"] = X_new["li_frac"].values
    return out.sort_values(["p_target", "pred_log_sigma"], ascending=False).reset_index(drop=True)


def rationale(r: pd.Series) -> str:
    return (f"{r.li_frac:.0%} Li; predicted log10 sigma {r.pred_log_sigma:.2f} +/- {r.uncertainty:.2f} "
            f"(P(top-5%) {r.p_target:.0%}); nearest measured OBELiX material {r.nearest_obelix} "
            f"({r.nearest_family}, log10 sigma {r.nearest_log_sigma}); E_hull {r.e_hull:.3f} eV/atom, "
            f"band gap {r.band_gap:.1f} eV{'; theoretical (not yet synthesized)' if r.theoretical else ''}.")


def shortlist(ranked: pd.DataFrame, n_in: int = 10, n_out: int = 5) -> pd.DataFrame:
    inside = ranked[ranked.in_domain].head(n_in).copy()
    outside = ranked[~ranked.in_domain].head(n_out).copy()
    inside["rank"], outside["rank"] = range(1, len(inside) + 1), range(1, len(outside) + 1)
    sl = pd.concat([inside, outside], ignore_index=True)
    sl["rationale"] = sl.apply(rationale, axis=1)
    sl["next_experiment"] = [NEXT_EXPERIMENT if (r.in_domain and r["rank"] <= 3) else None for _, r in sl.iterrows()]
    return sl


def publish(sl: pd.DataFrame, stats: dict) -> None:
    from integrations.supabase_sync import db

    if not db.enabled:
        print("Supabase not configured: skipped publishing")
        return
    db.delete("candidates", {"is_demo": "eq.false"})
    db.insert("candidates", [{
        "id": r.mp_id, "formula": r.formula, "mp_id": r.mp_id, "e_hull": round(float(r.e_hull), 4),
        "band_gap": round(float(r.band_gap), 2), "pred_log_sigma": round(float(r.pred_log_sigma), 2),
        "uncertainty": round(float(r.uncertainty), 2), "in_domain": bool(r.in_domain),
        "domain_distance": round(float(r.domain_distance), 2), "rationale": r.rationale, "rank": int(r["rank"]),
        "next_experiment": r.next_experiment, "is_demo": False} for _, r in sl.iterrows()])
    top3 = sl[sl.in_domain].head(3)
    db.upsert("runs", [{"id": DISCOVERY_RUN, "strategy": "discovery", "seed": 0, "budget": 3,
                        "status": "awaiting_approval", "is_demo": False}])
    db.insert("approvals", {
        "run_id": DISCOVERY_RUN, "round": 0, "design": "next_experiment", "status": "pending",
        "request": (f"Approve the next experiment for the top 3 new candidates "
                    f"({', '.join(f'{r.formula} [{r.mp_id}]' for _, r in top3.iterrows())}): {NEXT_EXPERIMENT}")})
    db.event(DISCOVERY_RUN, 0, "pi_director", "next_step",
             f"{stats['novel']} new Li insulators from Materials Project ranked; top 3 sent for approval.",
             {"top3": top3[["mp_id", "formula", "pred_log_sigma", "uncertainty"]].to_dict("records")})
    print(f"published {len(sl)} candidates, 1 pending approval for the top 3")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-query Materials Project (else use the cache)")
    ap.add_argument("--no-publish", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    mp = fetch_materials_project(args.refresh)
    new = novel(mp)
    ranked = rank(new)
    sl = shortlist(ranked)
    stats = {"queried": int(len(mp)), "novel": int(len(new)), "in_domain": int(ranked.in_domain.sum()),
             "criteria": {k: list(v) if isinstance(v, tuple) else v for k, v in CRITERIA.items()},
             "target_log_sigma": threshold("main"), "generated_at": datetime.now(timezone.utc).isoformat(),
             "label": "AI-generated hypotheses, not lab-validated"}
    OUT.mkdir(exist_ok=True)
    cols = ["rank", "in_domain", "mp_id", "formula", "pred_log_sigma", "uncertainty", "p_target", "e_hull",
            "band_gap", "domain_distance", "nearest_obelix", "nearest_family", "theoretical", "rationale",
            "next_experiment"]
    (OUT / "candidates.json").write_text(json.dumps({**stats, "candidates": json.loads(sl[cols].to_json(orient="records"))},
                                                    indent=1))
    print(f"Materials Project: {stats['queried']} Li insulators near the hull -> {stats['novel']} not in OBELiX "
          f"-> {stats['in_domain']} inside the model's domain ({time.time() - t0:.0f}s)")
    print(sl[cols[:11]].to_string(index=False))
    if not args.no_publish:
        publish(sl, stats)


if __name__ == "__main__":
    main()
