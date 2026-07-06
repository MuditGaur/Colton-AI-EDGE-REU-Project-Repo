# ===== CELL 1 — Convergence curves (eval reward vs steps), faceted by team, colored by radius =====
# Robust file finding: unzip 3090_train_results.zip + a100_train_results.zip and upload the curve JSONs
# (named <team>_<radius>_<seed>.json) anywhere into Colab — this searches recursively and parses names.
import json, glob, re
from collections import defaultdict
import numpy as np, matplotlib.pyplot as plt

# --- find + group the curve files (works whether they're in curves/, loose, or nested) ---
pat = re.compile(r"(?:^|/)(\d+)_([0-9.]+)_(\d+)\.json$")
files = [f for f in glob.glob("**/*.json", recursive=True) if pat.search(f)]
print(f"found {len(files)} curve JSON files.  sample: {files[:3]}")
if not files:
    print("!! none found. Upload the <team>_<radius>_<seed>.json files (unzip them from the "
          "train_results zips first). They can be loose or in a folder — this cell searches recursively.")

groups = defaultdict(list)
for f in files:
    m = pat.search(f); groups[(int(m.group(1)), float(m.group(2)))].append(f)

fig, axes = plt.subplots(1, len(TEAMS), figsize=(15, 4.2), sharex=True)
for ax, n in zip(axes, TEAMS):
    for r in RADII:
        fs = groups.get((n, r), [])
        if not fs:
            continue
        curves, xs = [], None
        for f in fs:
            d = json.load(open(f))
            xs = [p["step"] for p in d]
            curves.append([(p["eval"] if p["eval"] is not None else np.nan) for p in d])
        L = min(len(c) for c in curves)                       # align lengths across seeds
        Y = np.array([c[:L] for c in curves]); X = np.array(xs[:L])
        mean, std = np.nanmean(Y, axis=0), np.nanstd(Y, axis=0)
        ax.plot(X, mean, lw=1.7, label=f"train ρ={r}")
        ax.fill_between(X, mean - std, mean + std, alpha=0.18)
    ax.axhline(RAND(n, 1.2, "reward"), ls="--", c="gray", lw=1, alpha=0.7)
    ax.set_title(f"N = {n}"); ax.set_xlabel("environment steps")
axes[0].set_ylabel("eval episode return")
axes[0].legend(title="(dashed = random)", fontsize=9)
fig.suptitle("Convergence: eval reward vs. steps (mean ± std over 3 seeds)", y=1.03)
plt.savefig("fig1_convergence.png", bbox_inches="tight"); plt.show()
