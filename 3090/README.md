# 3090 pipeline — N=6 configs

Runs the light **N=6** half of the study on a rented RTX 3090. (The `a100/` dir runs N=9 + N=12,
identical cells except `TEAM_SIZES`.) Study: teams {6,9,12} × radii {0.3,0.7,1.2} × 3 seeds, InforMARL
(`network=rnn_graph`), 8 envs, sequential seeds, episode length 26.

## Run order (paste each cell into Jupyter, in order)
1. **cell1_setup.py** — clone Mava + `uv sync`, write the eval module, patch `rec_mappo` with all pipeline
   hooks (step-0 baseline, checkpoints, model-load, custom-eval). Run once. Expect `pipeline patches applied: True`.
2. **cell2_compile.py** — compile N=6 × 3 radii @ window=10, confirm each `OK` (not `OOM`), zip the cache.
   → download **`3090_cache.zip`**. (One compile per team×radius; radius is baked in.)
3. **cell3_train.py** — train N=6 × 3 radii × 3 seeds (9 runs) to ~2M each. Each run: a step-0 (untrained)
   eval point, eval every ~10k steps (window=10), a checkpoint every 200k + a final one, and a convergence
   JSON. → download **`3090_train_results.zip`** (models + curves).
4. **cell4_evaluate.py** — load each trained model, evaluate **zero-shot** at all 3 radii (150 episodes)
   with the validated handwritten eval (reward + collision_rate + goal_completion), plus a random baseline.
   → download **`3090_eval_results.csv`**.

## Settings (locked)
- window=10, `num_evaluation=200` (~2.05M steps), `XLA_PYTHON_CLIENT_PREALLOCATE=false` (cuSolver fix),
  compile cache at `/workspace/jax_cache`.
- Eval metrics: `reward` (mean episode return), `collision_rate` (agent pairs <0.30, summed/episode),
  `goal_completion` (fraction of landmarks with nearest agent <0.20 at episode end).

## To resume on a fresh 3090 (e.g. after a crash)
Run cell1, upload + unzip `3090_cache.zip` to `/workspace/jax_cache` (skips compile), then continue.
Caches are 3090-specific — do not reuse on the A100.

## Analysis
All analysis (H1/H2/H3 stats, plots) runs **locally on your Mac** from `3090_eval_results.csv` +
`a100_eval_results.csv` + the convergence JSONs. No JAX needed for that step.
