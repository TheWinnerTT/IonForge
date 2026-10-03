"""Agent-free strategies IonForge must beat.

All strategies share the same loop: pick a batch of unmeasured candidates, pay
for them through the Oracle, learn, repeat. Baselines can run past the agents'
budget so that "measurements to the k-th hit" is uncensored for them.
"""
from __future__ import annotations

import numpy as np

from lab.oracle import Oracle
from lab.surrogate import Surrogate

# Family priority a solid-state chemist would use before seeing any data: sulfides
# first, then halides, antiperovskites, garnets, NASICONs, perovskites, LISICONs.
# Fixed a priori (textbook ordering, e.g. Bachman et al., Chem. Rev. 2016,
# doi:10.1021/acs.chemrev.5b00563), never tuned on OBELiX.
EXPERT_TIERS: list[tuple[int, tuple[str, ...]]] = [
    (0, ("lgps", "argyrodite", "thio", "sulfide", "sulphide")),
    (1, ("halide", "chloride", "bromide", "iodide", "fluoride")),
    (2, ("antiperovskite", "anti-perovskite")),
    (3, ("garnet",)),
    (4, ("nasicon",)),
    (5, ("perovskite",)),
    (6, ("lisicon",)),
]


def expert_tier(family: str) -> int:
    f = str(family).lower()
    for tier, keys in EXPERT_TIERS:
        if any(k in f for k in keys):
            return tier
    return len(EXPERT_TIERS)


class Strategy:
    name = "base"

    def select(self, oracle: Oracle, rng: np.random.Generator, k: int) -> list[str]:
        raise NotImplementedError


class RandomStrategy(Strategy):
    name = "random"

    def select(self, oracle, rng, k):
        pool = oracle.unmeasured
        return list(rng.choice(pool, size=min(k, len(pool)), replace=False))


class ExpertStrategy(Strategy):
    """Static family ranking, random order within a family."""
    name = "heuristic"

    def select(self, oracle, rng, k):
        pub = oracle.public_table().loc[oracle.unmeasured]
        order = sorted(pub.index, key=lambda c: (expert_tier(pub.at[c, "family"]), rng.random()))
        return order[:k]


class BOStrategy(Strategy):
    """Random-forest surrogate + acquisition. Optional expert warm start for round 1.

    acquisition: "ucb" (mu + beta*sd, the pre-registered baseline) or "p_hit"
    (probability of clearing the target threshold). Both are kept so IonForge is
    compared against the strongest BO on each metric, not a weak one.
    """

    def __init__(self, beta: float = 1.0, warm_start: str = "random", n_init: int = 5,
                 acquisition: str = "ucb"):
        self.beta = beta
        self.warm_start = warm_start
        self.n_init = n_init
        self.acquisition = acquisition
        self.name = ("bo_cold" if warm_start == "random" else "bo_prior") + ("" if acquisition == "ucb" else "_phit")
        self._init = RandomStrategy() if warm_start == "random" else ExpertStrategy()

    def select(self, oracle, rng, k):
        if len(oracle.measured) < self.n_init:
            return self._init.select(oracle, rng, k)
        ids = list(oracle.measured)
        y = [oracle.measured[c].log_sigma for c in ids]
        sur = Surrogate(seed=int(rng.integers(1 << 31))).fit(ids, y)
        scores = sur.score(oracle.unmeasured, oracle.threshold, self.beta)
        return list(scores.sort_values(self.acquisition, ascending=False).index[:k])


# Names match the `runs.strategy` / `curves.strategy` values in supabase/schema.sql.
STRATEGIES = {
    "random": RandomStrategy,
    "heuristic": ExpertStrategy,
    "bo_cold": lambda: BOStrategy(warm_start="random"),
    "bo_prior": lambda: BOStrategy(warm_start="expert"),
    "bo_prior_phit": lambda: BOStrategy(warm_start="expert", acquisition="p_hit"),
}


def run_campaign(strategy: Strategy, task: str, seed: int, budget: int = 50, batch: int = 5) -> dict:
    """Run one campaign; returns the hit sequence in measurement order."""
    oracle = Oracle(task, budget=budget, seed=seed, record=False)
    rng = np.random.default_rng(seed)
    while oracle.remaining >= 1 and oracle.unmeasured:
        ids = strategy.select(oracle, rng, int(min(batch, oracle.remaining)))
        oracle.measure(ids, requested_by=strategy.name)
    seq = [m.is_hit for m in oracle.measured.values()]
    fams = [m.family if m.is_hit else None for m in oracle.measured.values()]
    return {
        "strategy": strategy.name,
        "task": task,
        "seed": seed,
        "budget": budget,
        "hits": seq,
        "hit_families": fams,
        "ids": list(oracle.measured),
        "total_hits_in_pool": oracle.total_hits_in_pool(),
    }
