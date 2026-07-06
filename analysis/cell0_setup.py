# ===== CELL 0 — SETUP (run first). Upload all_eval_results.csv to Colab, then run this. =====
# Loads the data + defines helpers used by every figure cell. matplotlib + pandas + numpy only.
import pandas as pd, numpy as np, matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 200, "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.3, "figure.autolayout": True})

df = pd.read_csv("all_eval_results.csv")            # <- upload this file to Colab first
df["team"] = df["team"].astype(int)

trained = df[df["is_random"] == 0].copy()
trained["train_radius"] = trained["train_radius"].astype(float)
trained["test_radius"]  = trained["test_radius"].astype(float)
rnd = df[df["is_random"] == 1].copy()
rnd["test_radius"] = rnd["test_radius"].astype(float)

TEAMS  = [6, 9, 12]
RADII  = [0.3, 0.7, 1.2]
METRICS = ["reward", "collision_rate", "goal_completion"]
COL = {6: "#1f77b4", 9: "#ff7f0e", 12: "#d62728"}   # per-team colors
A = 4.0                                              # world area = 2x2

def connectivity(N, rho):                            # expected neighbors per agent (theory: (N-1)*pi*rho^2/A)
    return (N - 1) * np.pi * rho**2 / A

def M(team, tr, te, metric):                         # mean over seeds, trained model
    d = trained[(trained.team == team) & np.isclose(trained.train_radius, tr) & np.isclose(trained.test_radius, te)]
    return d[metric].mean()

def S(team, tr, te, metric):                         # std over seeds
    d = trained[(trained.team == team) & np.isclose(trained.train_radius, tr) & np.isclose(trained.test_radius, te)]
    return d[metric].std()

def RAND(team, te, metric):                          # random baseline (mean over seeds)
    d = rnd[(rnd.team == team) & np.isclose(rnd.test_radius, te)]
    return d[metric].mean()

print(f"loaded {len(df)} rows | teams {TEAMS} | radii {RADII}")
print("random-baseline reward per team:", {n: round(RAND(n, 1.2, 'reward'), 1) for n in TEAMS})
