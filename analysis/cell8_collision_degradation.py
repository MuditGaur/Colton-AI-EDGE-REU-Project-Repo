# ===== CELL 8 — Collision-rate degradation (the effect in its most SENSITIVE metric) =====
# Same layout as cell2 (reward) but for collisions: the narrow-trained line's explosion going wide is
# far more dramatic here than in reward. This is arguably the stronger version of the core figure.
import numpy as np, matplotlib.pyplot as plt

fig, axes = plt.subplots(1, len(TEAMS), figsize=(15, 4.2), sharex=True)
for ax, n in zip(axes, TEAMS):
    for tr in RADII:
        y = [M(n, tr, te, "collision_rate") for te in RADII]
        e = [S(n, tr, te, "collision_rate") for te in RADII]
        ax.errorbar(RADII, y, yerr=e, marker="o", capsize=3, lw=1.8, label=f"train ρ={tr}")
    ax.plot(RADII, [RAND(n, te, "collision_rate") for te in RADII], "k--", lw=1, alpha=0.6, label="random")
    ax.set_title(f"N = {n}"); ax.set_xlabel("TEST radius"); ax.set_xticks(RADII)
axes[0].set_ylabel("collisions per episode (mean ± std)")
axes[0].legend(fontsize=9)
fig.suptitle("Zero-shot collisions vs test radius — narrow-trained policies explode when tested wide", y=1.03)
plt.savefig("fig8_collision_degradation.png", bbox_inches="tight"); plt.show()

for n in TEAMS:
    print(f"N={n}: train0.3 collisions  test0.3 -> test1.2 : "
          f"{M(n,0.3,0.3,'collision_rate'):.1f} -> {M(n,0.3,1.2,'collision_rate'):.1f}")
