"""Descriptors for every candidate.

Three named feature sets, so the surrogate choice is an explicit, documented
comparison (see `python -m lab.compare_features`):

  basic          Magpie-style stats over 8 pymatgen element properties + chemistry/cell
  matminer       matminer ElementProperty(magpie) + Stoichiometry + ValenceOrbital
  matminer_chem  matminer + the Li / anion / cell descriptors matminer does not provide

matminer pins: pymatgen==2026.3.23 and pandas<3 (newer pymatgen renamed an internal
used by matminer 0.10.1). Features are cached per formula, so the cache survives
changes to how the pool is built or indexed.
"""
from __future__ import annotations

import sys
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element

from lab.data import ROOT, load_pool

CACHE = ROOT / "data" / "cache"
DEFAULT_SET = "matminer_chem"
ANIONS = ("O", "S", "Se", "F", "Cl", "Br", "I", "N", "P", "B", "Si", "Ge")
PROPS = {
    "X": lambda e: e.X,
    "radius": lambda e: e.atomic_radius,
    "Z": lambda e: e.Z,
    "mass": lambda e: e.atomic_mass,
    "row": lambda e: e.row,
    "group": lambda e: e.group,
    "mendeleev": lambda e: e.mendeleev_no,
    "ionic_radius": lambda e: e.average_ionic_radius,
}


def _comp(formula: str) -> Composition:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Composition(formula)


def elemental_stats(formula: str) -> dict[str, float]:
    """The 'basic' set's element statistics (kept for the comparison)."""
    fracs = _comp(formula).fractional_composition.get_el_amt_dict()
    out: dict[str, float] = {}
    for name, fn in PROPS.items():
        vals, w = [], []
        for el, f in fracs.items():
            v = fn(Element(el))
            if v is not None:
                vals.append(float(v))
                w.append(f)
        vals, w = np.array(vals), np.array(w) / np.sum(w)
        mean = float((vals * w).sum())
        out[f"{name}_mean"] = mean
        out[f"{name}_std"] = float(np.sqrt((w * (vals - mean) ** 2).sum()))
        out[f"{name}_min"] = float(vals.min())
        out[f"{name}_max"] = float(vals.max())
    return out


def chemistry(formula: str) -> dict[str, float]:
    """Li-conductor chemistry a solid-state chemist reasons with (not in matminer)."""
    comp = _comp(formula)
    fracs = comp.fractional_composition.get_el_amt_dict()
    out = {
        "n_elements": float(len(fracs)),
        "li_frac": float(fracs.get("Li", 0.0)),
        "is_sulfide": float(fracs.get("S", 0) + fracs.get("Se", 0) > 0),
        "is_halide": float(sum(fracs.get(h, 0) for h in ("F", "Cl", "Br", "I")) > 0),
        "is_oxide": float(fracs.get("O", 0) > 0),
        "n_atoms_fu": float(comp.num_atoms),
    }
    for an in ANIONS:
        out[f"frac_{an}"] = float(fracs.get(an, 0.0))
    return out


def matminer_features(formulas: list[str]) -> pd.DataFrame:
    from matminer.featurizers.base import MultipleFeaturizer
    from matminer.featurizers.composition import ElementProperty, Stoichiometry, ValenceOrbital

    feat = MultipleFeaturizer([
        ElementProperty.from_preset("magpie"),
        Stoichiometry(),
        ValenceOrbital(props=["frac"]),
    ])
    df = pd.DataFrame({"formula": formulas})
    df["composition"] = [_safe_comp(f) for f in formulas]
    ok = df["composition"].notna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = feat.featurize_dataframe(df[ok].copy(), "composition", ignore_errors=True, pbar=False)
    out = out.drop(columns=["composition"]).set_index("formula")
    out.columns = [f"mm_{c}" for c in out.columns]
    return out.reindex(formulas)


def _safe_comp(formula: str):
    try:
        return _comp(formula)
    except Exception:
        return None


@lru_cache(maxsize=1)
def _formula_table() -> pd.DataFrame:
    """All descriptors per unique formula, cached to disk."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "features_by_formula.csv"
    formulas = sorted(set(load_pool()["composition"]))
    if path.exists():
        cached = pd.read_csv(path, index_col="formula")
        if set(formulas) <= set(cached.index):
            return cached
    basic, chem = [], []
    for f in formulas:
        try:
            basic.append({"formula": f, **{f"b_{k}": v for k, v in elemental_stats(f).items()}})
            chem.append({"formula": f, **chemistry(f)})
        except Exception as exc:  # unparsable formula: keep the row, impute later
            print(f"[features] {f!r}: {exc}", file=sys.stderr)
            basic.append({"formula": f})
            chem.append({"formula": f})
    table = pd.concat([
        pd.DataFrame(basic).set_index("formula"),
        pd.DataFrame(chem).set_index("formula"),
        matminer_features(formulas),
    ], axis=1)
    table.index.name = "formula"
    table.to_csv(path)
    return table


@lru_cache(maxsize=4)
def feature_table(feature_set: str = DEFAULT_SET) -> pd.DataFrame:
    """Feature matrix indexed by candidate_id (never contains the target)."""
    pool = load_pool().set_index("candidate_id")
    ft = _formula_table().reindex(pool["composition"]).set_index(pool.index)
    cell = pd.DataFrame(index=pool.index)
    n_atoms_cell = pool["Z"].astype(float) * ft["n_atoms_fu"]
    cell["space_group"] = pool["space_group"].astype(float)
    cell["vol_per_atom"] = pool["volume"] / n_atoms_cell
    cell["li_density"] = ft["li_frac"] / cell["vol_per_atom"]

    chem_cols = [c for c in ft.columns if not c.startswith(("b_", "mm_"))]
    basic_cols = [c for c in ft.columns if c.startswith("b_")]
    mm_cols = [c for c in ft.columns if c.startswith("mm_")]
    if feature_set == "basic":
        out = pd.concat([ft[basic_cols + chem_cols], cell], axis=1)
    elif feature_set == "matminer":
        out = ft[mm_cols].copy()
    elif feature_set == "matminer_chem":
        out = pd.concat([ft[mm_cols + chem_cols], cell], axis=1)
    else:
        raise KeyError(f"unknown feature set {feature_set!r}")
    out = out.loc[:, out.notna().any()]  # drop all-NaN columns
    return out.fillna(out.median(numeric_only=True))


def anonymized_descriptor_view(ids: list[str]) -> list[dict]:
    """What Ablation 2 shows the LLM: rounded descriptors, no formula, no family."""
    keep = ["li_frac", "is_sulfide", "is_halide", "is_oxide", "vol_per_atom", "li_density",
            "n_elements", "space_group", "mm_MagpieData mean Electronegativity",
            "mm_MagpieData mean CovalentRadius", "mm_frac p valence electrons"]
    ft = feature_table()
    ft = ft.loc[ids, [c for c in keep if c in ft.columns]].round(3)
    ft.columns = [c.replace("mm_MagpieData ", "").replace("mm_", "") for c in ft.columns]
    return [{"candidate_id": i, **row} for i, row in ft.to_dict("index").items()]


if __name__ == "__main__":
    for name in ("basic", "matminer", "matminer_chem"):
        print(f"{name:14s} {feature_table(name).shape}")
