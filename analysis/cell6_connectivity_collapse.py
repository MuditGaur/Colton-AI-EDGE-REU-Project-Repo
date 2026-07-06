# ===== CELL 6 — THEORY TEST: is the governing variable radius, or connectivity? =====
# Plot normalized performance retention vs (a) radius shift and (b) connectivity shift (expected-neighbor
# change, from theory (N-1)*pi*rho^2/A). If the per-team curves COLLAPSE onto one line under connectivity
# but not under radius, then graph connectivity — not radius — governs generalization.
import numpy as np, matplotlib.pyplot as plt

def retention(team, tr, te):   # fraction of "learnable performance" kept: 1.0 at matched, 0.0 at random
    base = M(team, tr, tr, "reward"); r = RAND(team, te, "reward")
    return (M(team, tr, te, "reward") - r) / (base - r + 1e-9)

pts = []  # (team, dradius, dconn, retention)
for n in TEAMS:
    for tr in RADII:
        for te in RADII:
            pts.append((n, te - tr, connectivity(n, te) - connectivity(n, tr), retention(n, tr, te)))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
for ax, xi, xlab in [(a1, 1, "Δ radius  (test − train)"), (a2, 2, "Δ connectivity  (test − train expected neighbors)")]:
    for n in TEAMS:
        xs = [p[xi] for p in pts if p[0] == n]; ys = [p[3] for p in pts if p[0] == n]
        ax.scatter(xs, ys, s=70, color=COL[n], label=f"N={n}", alpha=0.85, edgecolor="k", linewidth=0.4)
    ax.axhline(1, ls=":", c="gray"); ax.axhline(0, ls="--", c="gray", alpha=0.6)
    ax.set_xlabel(xlab); ax.set_ylabel("performance retention  (1=matched, 0=random)"); ax.legend()
a1.set_title("vs RADIUS shift"); a2.set_title("vs CONNECTIVITY shift")
fig.suptitle("Do the team curves collapse under connectivity? (if a2 lines up and a1 doesn't → connectivity governs)", y=1.03)
plt.savefig("fig6_connectivity_collapse.png", bbox_inches="tight"); plt.show()
