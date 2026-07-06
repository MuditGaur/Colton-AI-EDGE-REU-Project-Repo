# ===== CELL 5 — "Reward masks, collisions reveal": same shift, two metrics =====
# For the narrow->wide shift, show how little reward moves vs how much collisions explode, across teams.
# Both expressed as % change from the matched (train==test) baseline so they're comparable.
import numpy as np, matplotlib.pyplot as plt

def pct_change(team, tr, te, metric):
    base = M(team, tr, tr, metric)
    return 100.0 * (M(team, tr, te, metric) - base) / abs(base)

reward_chg = [pct_change(n, 0.3, 1.2, "reward") for n in TEAMS]          # narrow->wide reward change
collis_chg = [pct_change(n, 0.3, 1.2, "collision_rate") for n in TEAMS]  # narrow->wide collision change

x = np.arange(len(TEAMS)); w = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - w/2, reward_chg, w, label="reward change", color="#1f77b4")
ax.bar(x + w/2, collis_chg, w, label="collision change", color="#d62728")
ax.axhline(0, c="k", lw=0.8)
ax.set_xticks(x, [f"N={n}" for n in TEAMS]); ax.set_ylabel("% change from matched baseline")
ax.set_title("narrow→wide shift (train 0.3 → test 1.2): reward barely moves, collisions explode")
ax.legend()
for i in range(len(TEAMS)):
    ax.text(i - w/2, reward_chg[i], f"{reward_chg[i]:+.0f}%", ha="center", va="bottom", fontsize=9)
    ax.text(i + w/2, collis_chg[i], f"{collis_chg[i]:+.0f}%", ha="center", va="bottom", fontsize=9)
plt.savefig("fig5_reward_masks_collisions.png", bbox_inches="tight"); plt.show()
print("reward %chg:", [round(v) for v in reward_chg], " | collision %chg:", [round(v) for v in collis_chg])
