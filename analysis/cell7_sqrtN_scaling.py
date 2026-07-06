# ===== CELL 7 — THEORY TEST: does reward scale like -sqrt(N)? =====
# Reward = -sum over N landmarks of nearest-agent distance; nearest-of-N distance ~ 0.5*sqrt(A/N),
# so reward ~ -0.5*sqrt(A*N) ~ -sqrt(N). Test on the random baseline (cleanest, policy-independent) and
# on the matched trained reward. Overlay a fitted power law to read off the empirical exponent.
import numpy as np, matplotlib.pyplot as plt

Ns = np.array(TEAMS, float)
rand_r    = np.array([RAND(n, 1.2, "reward") for n in TEAMS])                 # random baseline
matched_r = np.array([np.mean([M(n, r, r, "reward") for r in RADII]) for n in TEAMS])  # avg matched trained

def fit_power(N, y):                     # fit |y| = c * N^p  (log-log)
    p, logc = np.polyfit(np.log(N), np.log(np.abs(y)), 1)
    return p, np.exp(logc)

fig, ax = plt.subplots(figsize=(8, 5.2))
for y, lab, c in [(rand_r, "random baseline", "gray"), (matched_r, "trained (matched radius)", "#1f77b4")]:
    p, coef = fit_power(Ns, y)
    ax.plot(Ns, y, "o", color=c, ms=9, label=f"{lab}: |reward| ∝ N^{p:.2f}")
    xx = np.linspace(6, 12, 50); ax.plot(xx, -coef * xx**p, "-", color=c, lw=1.6, alpha=0.7)
# sqrt(N) reference anchored at N=6 random
ref = rand_r[0] * np.sqrt(Ns / 6)
ax.plot(Ns, ref, "k--", lw=1.2, alpha=0.6, label="√N reference (theory)")
ax.set_xticks(TEAMS); ax.set_xlabel("team size N"); ax.set_ylabel("episode return")
ax.set_title("Reward scaling with team size vs. the √N prediction"); ax.legend(fontsize=9)
plt.savefig("fig7_sqrtN_scaling.png", bbox_inches="tight"); plt.show()
print("random reward:", [round(v,1) for v in rand_r], " matched trained:", [round(v,1) for v in matched_r])
