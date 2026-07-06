# analysis — Colab visualization cells

Local (Mac) analysis of `model_data/all_eval_results.csv`. Paste each cell into a Google Colab notebook,
in order. **matplotlib + pandas + numpy only** (all pre-installed in Colab — no `pip install` needed).

## What to upload to Colab
- **`all_eval_results.csv`** (from `model_data/`) — needed by every cell.
- **For cell1 only:** the convergence curves. Unzip `3090_train_results.zip` + `a100_train_results.zip`,
  and upload the `curves/` folder (files named `<team>_<radius>_<seed>.json`).

## Run order
- **cell0_setup.py** — run FIRST. Loads the CSV, defines helpers (`M`, `RAND`, `connectivity`) that every
  other cell uses. (Colab keeps this state across cells.)
- **cell1_convergence.py** — eval reward vs steps, per team, per radius (needs the `curves/` folder).
- **cell2_degradation_H1.py** — reward vs test radius, one line per train radius, faceted by team. **H1/H2 core.**
- **cell3_heatmaps.py** — train×test heatmaps for all 3 metrics × 3 teams (the whole dataset at a glance).
- **cell4_asymmetry_H2_H3.py** — **HEADLINE:** the directional-fragility collapse (narrow→wide vs wide→narrow
  across N). Small teams asymmetric → large teams symmetric.
- **cell5_reward_vs_collisions.py** — "reward masks, collisions reveal": same shift, both metrics.
- **cell6_connectivity_collapse.py** — theory test: does degradation collapse across teams vs Δ-connectivity
  (but not vs Δ-radius)? → evidence connectivity, not radius, governs generalization.
- **cell7_sqrtN_scaling.py** — theory test: does reward scale like −√N? Fits the empirical exponent.
- **cell8_collision_degradation.py** — the core effect in its most SENSITIVE metric (collisions vs test
  radius, per train radius, per team). Stronger than the reward version.
- **cell9_redundancy.py** — the mechanism behind "bigger = more robust": reward retention under the
  narrow→wide shift rises with N, and per-pair collision load falls with N.
- **cell10_retention_heatmap.py** — normalized retention (0–1) train×test grid per team; apples-to-apples
  across team sizes (raw reward isn't).
- **cell11_coverage_curves.py** — goal completion vs test radius; the coverage side of the
  coverage↔collision tradeoff.

Each cell **saves a PNG** (`fig1_...png` … `fig7_...png`) *and* displays it inline. After running, download
the PNGs from Colab and drop them into the `figures/` folder.

## Metric definitions (reminder — matches the eval)
- `reward` = mean episode return (sum over 25 steps, mean over agents).
- `collision_rate` = agent pairs <0.30 apart, summed over the episode (mean over episodes).
- `goal_completion` = fraction of landmarks with nearest agent <0.20 at episode end.
- `is_random=1` rows = untrained baseline; `train_radius="random"` in the CSV.
