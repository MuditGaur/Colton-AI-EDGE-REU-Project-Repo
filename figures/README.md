# figures — generated visualizations

Drop the PNGs you download from Colab here (one per analysis cell). Suggested names match the cells:

- `fig1_convergence.png`            — convergence curves (reward vs steps, per team/radius)
- `fig2_degradation_reward.png`     — H1/H2: reward vs test radius, per train radius, per team
- `fig3_heatmaps.png`               — train×test matrices, all metrics × teams
- `fig4_asymmetry_collapse.png`     — HEADLINE: directional fragility collapses with team size (H2→H3)
- `fig5_reward_masks_collisions.png`— reward vs collision change for the narrow→wide shift
- `fig6_connectivity_collapse.png`  — theory: retention vs Δradius vs Δconnectivity
- `fig7_sqrtN_scaling.png`          — theory: reward scaling vs √N
- `fig8_collision_degradation.png`  — the core effect in collisions (the sensitive metric)
- `fig9_redundancy.png`             — why bigger teams are more robust (retention up, per-pair collisions down)
- `fig10_retention_heatmap.png`     — normalized retention grid, comparable across team sizes
- `fig11_coverage_curves.png`       — coverage vs test radius (coverage↔collision tradeoff)

Keep the raw figures here; the paper-final versions (relabeled, cropped) can go in a `paper_figures/`
subfolder later.
