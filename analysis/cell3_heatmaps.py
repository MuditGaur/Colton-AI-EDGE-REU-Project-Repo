# ===== CELL 3 — Train x Test heatmaps for all 3 metrics x 3 teams (the full picture) =====
# Rows = metric (reward, collisions, goals); cols = team. Each cell = mean over seeds. Reveals H1/H2 at a glance.
import numpy as np, matplotlib.pyplot as plt

titles = {"reward": "reward (higher=better)", "collision_rate": "collisions (lower=better)",
          "goal_completion": "goal completion (higher=better)"}
cmaps  = {"reward": "viridis", "collision_rate": "magma_r", "goal_completion": "viridis"}

fig, axes = plt.subplots(len(METRICS), len(TEAMS), figsize=(13, 11))
for i, metric in enumerate(METRICS):
    for j, n in enumerate(TEAMS):
        ax = axes[i, j]
        Z = np.array([[M(n, tr, te, metric) for te in RADII] for tr in RADII])  # rows=train, cols=test
        im = ax.imshow(Z, cmap=cmaps[metric], aspect="auto")
        ax.set_xticks(range(3), RADII); ax.set_yticks(range(3), RADII)
        for a in range(3):
            for b in range(3):
                ax.text(b, a, f"{Z[a,b]:.1f}", ha="center", va="center",
                        color="white" if metric == "collision_rate" else "black", fontsize=9)
        if i == 0: ax.set_title(f"N = {n}")
        if j == 0: ax.set_ylabel(f"{titles[metric]}\n\nTRAIN radius")
        if i == len(METRICS)-1: ax.set_xlabel("TEST radius")
        plt.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle("Zero-shot train×test matrices (diagonal = matched; off-diagonal = radius shift)", y=1.0)
plt.savefig("fig3_heatmaps.png", bbox_inches="tight"); plt.show()
