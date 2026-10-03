"""Discovery metrics: measurements to the k-th hit, discovery curves, speedups."""
from __future__ import annotations

import numpy as np


def measurements_to_k(hits: list[bool], k: int) -> float:
    """1-based index of the k-th hit; inf if never reached (censored)."""
    idx = np.flatnonzero(np.asarray(hits, dtype=bool))
    return float(idx[k - 1] + 1) if len(idx) >= k else float("inf")


def discovery_curve(hits: list[bool], length: int) -> np.ndarray:
    h = np.asarray(hits[:length], dtype=float)
    c = np.cumsum(h)
    if len(c) < length:
        c = np.pad(c, (0, length - len(c)), mode="edge")
    return c


def summarize(campaigns: list[dict], ks=(3, 5, 10), horizon: int = 50) -> dict:
    out: dict = {"n_seeds": len(campaigns)}
    for k in ks:
        v = np.array([measurements_to_k(c["hits"], k) for c in campaigns])
        out[f"to_k{k}"] = _robust_stats(v)
    curves = np.stack([discovery_curve(c["hits"], horizon) for c in campaigns])
    out["curve_median"] = np.median(curves, 0).tolist()
    out["curve_q25"] = np.quantile(curves, 0.25, 0).tolist()
    out["curve_q75"] = np.quantile(curves, 0.75, 0).tolist()
    out[f"hits_at_{horizon}"] = _robust_stats(curves[:, -1])
    # War Room secondary metric: ten doped variants of one material are not ten discoveries.
    fams = np.array([distinct_families(c.get("hit_families", []), horizon) for c in campaigns], float)
    out[f"families_at_{horizon}"] = _robust_stats(fams)
    fcurves = np.stack([families_curve(c.get("hit_families", []), horizon) for c in campaigns])
    out["families_curve_median"] = np.median(fcurves, 0).tolist()
    out["families_curve_q25"] = np.quantile(fcurves, 0.25, 0).tolist()
    out["families_curve_q75"] = np.quantile(fcurves, 0.75, 0).tolist()
    return out


def families_curve(hit_families: list[str | None], length: int) -> np.ndarray:
    seen, out = set(), []
    for f in hit_families[:length]:
        if f:
            seen.add(f)
        out.append(len(seen))
    out += [len(seen)] * (length - len(out))
    return np.array(out, float)


def distinct_families(hit_families: list[str | None], horizon: int) -> int:
    return len({f for f in hit_families[:horizon] if f})


def random_expected_to_k(n_pool: int, n_hits: int, k: int) -> float:
    """Exact expectation for sampling without replacement (negative hypergeometric)."""
    return k * (n_pool + 1) / (n_hits + 1)


def speedup(baseline: list[float], method: list[float], n_boot: int = 2000, seed: int = 0) -> dict:
    """Ratio of median measurements-to-k with a bootstrap 90% interval."""
    b, m = np.asarray(baseline, float), np.asarray(method, float)
    rng = np.random.default_rng(seed)
    point = _safe_ratio(np.median(b), np.median(m))
    boots = [
        _safe_ratio(np.median(rng.choice(b, len(b))), np.median(rng.choice(m, len(m))))
        for _ in range(n_boot)
    ]
    lo, hi = np.quantile(boots, [0.05, 0.95], method="nearest")  # robust to inf (censored)
    return {"point": point, "ci90": [float(lo), float(hi)]}


def _safe_ratio(a: float, b: float) -> float:
    if np.isinf(b):
        return 0.0
    return float(a / b)


def _robust_stats(v: np.ndarray) -> dict:
    finite = np.isfinite(v)
    return {
        "median": float(np.median(v)),
        "q25": float(np.quantile(v, 0.25, method="nearest")),
        "q75": float(np.quantile(v, 0.75, method="nearest")),
        "frac_reached": float(finite.mean()),
        "values": [float(x) if np.isfinite(x) else None for x in v],
    }
