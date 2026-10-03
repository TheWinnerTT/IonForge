# OBELiX EDA (gate f0-7)

Reproduce: `python scripts/eda_obelix.py` → `data/pool.csv`, `results/eda_summary.json`.

| | |
|---|---|
| Materials in pool | **599** (37 have an upper bound `<1E-10` / `<1E-8`, kept at the bound) |
| With CIF | 321 |
| σ ≥ 10⁻³ S/cm | 100 (**16.7%**), above the ~10% limit, so random would find them too fast |
| **Target used** | **top 5% of log σ: log σ ≥ −2.316** (30 targets) |
| Unique reduced formulas | 508 |
| Near-duplicate groups (family + host chemical system, dopants < 4 at.% ignored) | 203, of which 74 have more than one member |
| Families | 36 after merging OBELiX label variants (case, plurals, typos: "Halide"/"halides", "argyrodite"/"argyrodites"); 43 raw labels kept in `family_obelix` |
| Families among targets | 6: LGPS 21, unknown 3, NASICON 2, argyrodites 2, thio-LISICON 1, antiperovskite 1 |

## Random baseline (budget 50, 2000 simulations)

| k | P(random finds k within 50) | median measurements when it does |
|---|---|---|
| 1 | 93% | 12 |
| 2 | 73% | 26 |
| **3** | **47%** | **34** |
| 4 | 23% | 39 |
| 5 | 11% | 41 |

**k = 3** (the script picks the smallest k whose random median is ≥ 60% of the budget). Random needs ~34 of the 50 measurements and fails half the time, which leaves room to show a speed-up. When a strategy does not reach k within the budget, record it as censored (51) and report the success rate next to the median.

## Notes
- 21 of the 30 targets are LGPS-type, so "superionic found" alone rewards hammering one family. That is why **distinct families found** is the secondary metric.
- `data/pool.csv` contains the hidden `log_sigma`. Only `lab/oracle.py` may read that column, and agents only see composition and descriptors.
- `group_id` is used by the no-leak filter (`literature/leak_filter.py`) to also block doped variants of pool materials.
