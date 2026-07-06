# ===== CELL 9 — The REDUNDANCY mechanism: why bigger teams are MORE robust to radius shift =====
# Left : reward retention under the narrow->wide shift, vs team size (rises = bigger teams keep more).
# Right: collisions PER PAIR (÷ C(N,2)) vs team size (falls = each pair collides less; load is spread).
# Together these explain the counterintuitive "bigger = more robust" result.
import numpy as np, matplotlib.pyplot as plt

def retention(n, tr, te):
    base = M(n, tr, tr, "reward"); r = RAND(n, te, "reward")
    return (M(n, tr, te, "reward") - r) / (base - r + 1e-9)

def pairs(n): return n * (n - 1) / 2.0

ret_nw    = [retention(n, 0.3, 1.2) for n in TEAMS]                       # narrow->wide retention
perpair_m = [np.mean([M(n, r, r, "collision_rate") for r in RADII]) / pairs(n) for n in TEAMS]  # matched
perpair_nw= [M(n, 0.3, 1.2, "collision_rate") / pairs(n) for n in TEAMS]  # narrow->wide
perpair_rd= [RAND(n, 1.2, "collision_rate") / pairs(n) for n in TEAMS]    # random

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.6))

axL.plot(TEAMS, ret_nw, "o-", lw=2.2, ms=10, color="#2ca02c")
axL.axhline(1, ls=":", c="gray", label="matched (no degradation)")
axL.axhline(0, ls="--", c="gray", alpha=0.6, label="random")
axL.set_ylim(-0.05, 1.1); axL.set_xticks(TEAMS)
axL.set_xlabel("team size N"); axL.set_ylabel("reward retention under narrow→wide shift")
axL.set_title("Bigger teams retain MORE under flooding"); axL.legend(fontsize=9)
for n, v in zip(TEAMS, ret_nw): axL.annotate(f"{v:.2f}", (n, v), textcoords="offset points", xytext=(0, 8), ha="center")

axR.plot(TEAMS, perpair_m,  "s-", lw=2, ms=8, label="matched (train=test)")
axR.plot(TEAMS, perpair_nw, "o-", lw=2, ms=8, label="narrow→wide (0.3→1.2)")
axR.plot(TEAMS, perpair_rd, "^--", lw=1.6, ms=7, color="gray", label="random")
axR.set_xticks(TEAMS); axR.set_xlabel("team size N"); axR.set_ylabel("collisions per agent-pair  (÷ C(N,2))")
axR.set_title("Per-pair collision load DROPS with team size"); axR.legend(fontsize=9)

fig.suptitle("Redundancy: more agents → more performance retained + lower per-pair collision burden", y=1.02)
plt.savefig("fig9_redundancy.png", bbox_inches="tight"); plt.show()
print("narrow→wide retention:", [round(v,2) for v in ret_nw])
print("per-pair collisions (matched):", [round(v,2) for v in perpair_m])
