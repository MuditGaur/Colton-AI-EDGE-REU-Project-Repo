# ===== CELL 11 — Coverage (goal completion) vs test radius: the other half of the tradeoff =====
# Narrow sensing -> agents can't see each other -> pile onto landmarks -> HIGH coverage (+ high collisions).
# Wide sensing -> agents spread to avoid each other -> LOWER coverage. This shows the coverage side of the
# coverage<->collision tradeoff that drives the whole phenomenon.
import numpy as np, matplotlib.pyplot as plt

fig, axes = plt.subplots(1, len(TEAMS), figsize=(15, 4.2), sharex=True)
for ax, n in zip(axes, TEAMS):
    for tr in RADII:
        y = [M(n, tr, te, "goal_completion") for te in RADII]
        e = [S(n, tr, te, "goal_completion") for te in RADII]
        ax.errorbar(RADII, y, yerr=e, marker="o", capsize=3, lw=1.8, label=f"train ρ={tr}")
    ax.plot(RADII, [RAND(n, te, "goal_completion") for te in RADII], "k--", lw=1, alpha=0.6, label="random")
    ax.set_title(f"N = {n}"); ax.set_xlabel("TEST radius"); ax.set_xticks(RADII); ax.set_ylim(0, None)
axes[0].set_ylabel("goal completion (fraction of landmarks covered)")
axes[0].legend(fontsize=9)
fig.suptitle("Coverage vs test radius — narrow-trained policies cover best (they cluster on landmarks)", y=1.03)
plt.savefig("fig11_coverage_curves.png", bbox_inches="tight"); plt.show()
