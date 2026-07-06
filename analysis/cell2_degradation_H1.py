# ===== CELL 2 — H1/H2: zero-shot reward vs TEST radius, one line per TRAIN radius, faceted by team =====
# The core degradation figure: narrow-trained lines fall off going wide; wide-trained lines stay flat.
import numpy as np, matplotlib.pyplot as plt

fig, axes = plt.subplots(1, len(TEAMS), figsize=(15, 4.2), sharex=True)
for ax, n in zip(axes, TEAMS):
    for tr in RADII:
        y  = [M(n, tr, te, "reward") for te in RADII]
        e  = [S(n, tr, te, "reward") for te in RADII]
        ax.errorbar(RADII, y, yerr=e, marker="o", capsize=3, lw=1.8, label=f"train ρ={tr}")
    ax.plot(RADII, [RAND(n, te, "reward") for te in RADII], "k--", lw=1, alpha=0.6, label="random")
    ax.axvspan(0, 0, alpha=0)  # keeps x tidy
    ax.set_title(f"N = {n}"); ax.set_xlabel("TEST radius"); ax.set_xticks(RADII)
axes[0].set_ylabel("episode return (mean ± std)")
axes[0].legend(fontsize=9)
fig.suptitle("Zero-shot generalization: reward vs test radius (each line = a training radius)", y=1.03)
plt.savefig("fig2_degradation_reward.png", bbox_inches="tight"); plt.show()

# quick text summary of the diagonal (matched train==test) + the narrow->wide drop
print("matched (train==test) reward:", {n: round(M(n, r, r, 'reward'), 1) for n in TEAMS for r in [0.3]})
for n in TEAMS:
    print(f"N={n}: train0.3 test0.3 -> test1.2 : {M(n,0.3,0.3,'reward'):.1f} -> {M(n,0.3,1.2,'reward'):.1f}")
