"""
Neighbor connectivity analysis for the Mava / JaxMARL `simple_spread` environment.

VERIFIED to match the original InforMARL setup:
  - JaxMARL simple_spread spawns agents uniform in [-1, +1]^2  (a 2x2 world)
    (jax.random.uniform(key, (num_agents, 2), minval=-1.0, maxval=+1.0))
  - Edges/neighbors use Euclidean distance <= visibility_radius (MPEGraphWrapper).
So this geometry is identical to the original analysis; radii should carry over.

Covers team sizes for both candidate studies: {4,6,8} and {6,9,12}.
Pure geometry — runs locally in seconds, no GPU.
"""
import numpy as np

WORLD = 2.0            # 2x2 world: agents uniform in [-1, 1]^2
N_EPISODES = 20000     # random initialisations per (N, radius)
AGENT_COUNTS = [4, 6, 8, 9, 12]
RADII = np.round(np.arange(0.1, 1.65, 0.05), 2)
CANDIDATE_RADII = [0.3, 0.7, 1.2]
rng = np.random.default_rng(0)


def avg_neighbors(n_agents, radius):
    pos = rng.uniform(-WORLD / 2, WORLD / 2, size=(N_EPISODES, n_agents, 2))
    dist = np.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1)  # (E,N,N)
    within = (dist <= radius) & ~np.eye(n_agents, dtype=bool)[None]
    return within.sum(axis=2).mean()   # mean neighbors per agent


results = {n: np.array([avg_neighbors(n, r) for r in RADII]) for n in AGENT_COUNTS}

# ---- full table ----
print("avg agent-neighbors per agent  (2x2 world, uniform spawn)\n")
header = "radius | " + " | ".join(f"N={n:>2}" for n in AGENT_COUNTS)
print(header); print("-" * len(header))
for i, r in enumerate(RADII):
    row = f"  {r:.2f} | " + " | ".join(f"{results[n][i]:>4.2f}" for n in AGENT_COUNTS)
    print(row)

# ---- connectivity at the candidate radii (the regimes that justify radius choices) ----
print("\n\n=== connectivity at candidate radii (avg neighbors per agent) ===")
print("radius | " + " | ".join(f"N={n:>2}" for n in AGENT_COUNTS))
for rho in CANDIDATE_RADII:
    idx = int(np.argmin(np.abs(RADII - rho)))
    print(f"  {rho:.1f}  | " + " | ".join(f"{results[n][idx]:>4.2f}" for n in AGENT_COUNTS))

# ---- save plot if matplotlib is available ----
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    for n in AGENT_COUNTS:
        plt.plot(RADII, results[n], marker="o", ms=3, label=f"N={n}")
    for rho in CANDIDATE_RADII:
        plt.axvline(rho, ls="--", c="gray", alpha=0.4)
    plt.axhline(1, ls=":", c="black", alpha=0.4)
    plt.xlabel("sensing radius  rho"); plt.ylabel("avg agent-neighbors per agent")
    plt.title("Mava/JaxMARL simple_spread — neighbor connectivity (2x2 world)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("neighbor_analysis_mava.png", dpi=150)
    print("\nsaved plot -> neighbor_analysis_mava.png")
except Exception as e:
    print(f"\n(plot skipped: {e})")
