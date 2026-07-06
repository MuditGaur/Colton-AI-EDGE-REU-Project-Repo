# a100 pipeline — N=9 and N=12 configs

Runs the dense **N=9 + N=12** half of the study on a rented A100 (needs the VRAM; N=12 at 1 seed is too big
for the 3090's 24 GB). Identical cells to `3090/` except `TEAM_SIZES = [9, 12]`. Study: teams {6,9,12} ×
radii {0.3,0.7,1.2} × 3 seeds, InforMARL (`network=rnn_graph`), 8 envs, sequential seeds.

## Run order (paste each cell into Jupyter, in order)
1. **cell1_setup.py** — clone Mava + `uv sync`, write eval module, patch `rec_mappo`. Expect `pipeline patches applied: True`.
2. **cell2_compile.py** — compile N=9 & N=12 × 3 radii @ window=10 (6 configs). **Watch for `OOM` on N=12** —
   if it appears, stop and tell Claude (shrink envs/rollout). Zip → download **`a100_cache.zip`**.
3. **cell3_train.py** — train N=9 & N=12 × 3 radii × 3 seeds (18 runs) to ~2M each: step-0 point, window=10,
   checkpoint every 200k + final, convergence JSON. → download **`a100_train_results.zip`**.
   (This is the long pole — ~a day of A100 time. Consider running N=9 and N=12 as separate sessions.)
4. **cell4_evaluate.py** — zero-shot eval each model @ all 3 radii (150 eps) + random baseline. → download
   **`a100_eval_results.csv`**.

## Settings (locked)
- window=10, `num_evaluation=200` (~2.05M steps), `XLA_PYTHON_CLIENT_PREALLOCATE=false`, cache at `/workspace/jax_cache`.
- Metrics: `reward`, `collision_rate` (pairs <0.30 summed/episode), `goal_completion` (landmarks with nearest agent <0.20).

## Notes
- Caches are A100-specific — reuse only on an A100. To resume: cell1, unzip `a100_cache.zip` to `/workspace/jax_cache`, continue.
- If N=12 training time is a concern, run the N=12 pilot (one seed to ~2-3M) first to confirm where it
  converges, then decide whether to shorten. `num_evaluation` in cell3 controls total steps.
- Analysis is local on the Mac from the CSVs + convergence JSONs.
