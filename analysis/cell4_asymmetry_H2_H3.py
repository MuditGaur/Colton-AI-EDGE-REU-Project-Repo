# ===== CELL 4 — HEADLINE: the asymmetry collapse (H2 -> H3) =====
# Collision fragility (relative to each team's random baseline) for narrow->wide vs wide->narrow, across N.
# Small teams: strongly asymmetric. Large teams: the two directions converge -> asymmetry dissolves with scale.
import numpy as np, matplotlib.pyplot as plt

def frac_over_random(team, tr, te, metric):
    return M(team, tr, te, metric) / RAND(team, te, metric)

narrow_wide = [frac_over_random(n, 0.3, 1.2, "collision_rate") for n in TEAMS]   # trained narrow, tested wide
wide_narrow = [frac_over_random(n, 1.2, 0.3, "collision_rate") for n in TEAMS]   # trained wide, tested narrow

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.6))

# left: the collapse
axL.plot(TEAMS, narrow_wide, "o-", lw=2.2, ms=9, color="#d62728", label="narrow→wide (train 0.3, test 1.2)")
axL.plot(TEAMS, wide_narrow, "s-", lw=2.2, ms=9, color="#1f77b4", label="wide→narrow (train 1.2, test 0.3)")
axL.axhline(1.0, ls="--", c="gray", lw=1, label="random baseline (=1.0)")
axL.set_xticks(TEAMS); axL.set_xlabel("team size N"); axL.set_ylabel("collisions ÷ random baseline")
axL.set_title("Directional fragility collapses with team size"); axL.legend(fontsize=9)

# right: the gap between the two directions (the "asymmetry index")
gap = [nw - wn for nw, wn in zip(narrow_wide, wide_narrow)]
axR.bar([str(n) for n in TEAMS], gap, color=[COL[n] for n in TEAMS])
axR.set_xlabel("team size N"); axR.set_ylabel("asymmetry index  (narrow→wide − wide→narrow)")
axR.set_title("Asymmetry shrinks as teams grow")
for i, n in enumerate(TEAMS): axR.text(i, gap[i], f"{gap[i]:.1f}", ha="center", va="bottom")

fig.suptitle("H2→H3: small teams are asymmetrically fragile; large teams are symmetrically fragile", y=1.02)
plt.savefig("fig4_asymmetry_collapse.png", bbox_inches="tight"); plt.show()
print("narrow→wide (×random):", [round(x,1) for x in narrow_wide])
print("wide→narrow (×random):", [round(x,1) for x in wide_narrow])
print("asymmetry index:", [round(g,1) for g in gap])
