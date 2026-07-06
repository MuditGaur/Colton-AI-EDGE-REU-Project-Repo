# ===== CELL 10 — Normalized retention heatmap (apples-to-apples across team sizes) =====
# retention = (reward - random) / (matched_reward - random): 1.0 = as good as at the training radius,
# 0.0 = no better than random. Unlike raw reward (which scales with N), this is comparable across teams.
# One 3x3 (train x test) panel per team. The bright diagonal = matched; dark off-diagonal = shift damage.
import numpy as np, matplotlib.pyplot as plt

def retention(n, tr, te):
    base = M(n, tr, tr, "reward"); r = RAND(n, te, "reward")
    return (M(n, tr, te, "reward") - r) / (base - r + 1e-9)

fig, axes = plt.subplots(1, len(TEAMS), figsize=(15, 4.6))
for ax, n in zip(axes, TEAMS):
    Z = np.array([[retention(n, tr, te) for te in RADII] for tr in RADII])   # rows=train, cols=test
    im = ax.imshow(Z, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(3), RADII); ax.set_yticks(range(3), RADII)
    for a in range(3):
        for b in range(3):
            ax.text(b, a, f"{Z[a,b]:.2f}", ha="center", va="center", fontsize=11,
                    color="black")
    ax.set_title(f"N = {n}"); ax.set_xlabel("TEST radius")
    if ax is axes[0]: ax.set_ylabel("TRAIN radius")
    plt.colorbar(im, ax=ax, fraction=0.046, label="retention")
fig.suptitle("Performance retention (1=matched, 0=random) — comparable across team sizes", y=1.02)
plt.savefig("fig10_retention_heatmap.png", bbox_inches="tight"); plt.show()
