import numpy as np
import matplotlib.pyplot as plt

# World is a 2x2 square (-1 to 1 in x and y)
WORLD_SIZE = 2.0
N_EPISODES = 2000

def random_positions(n, world_size=WORLD_SIZE):
    return np.random.uniform(-world_size/2, world_size/2, size=(n, 2))

def avg_agent_neighbors(n_agents, radius, n_episodes=N_EPISODES):
    neighbor_counts = []
    for _ in range(n_episodes):
        pos = random_positions(n_agents)
        for i in range(n_agents):
            count = 0
            for j in range(n_agents):
                if i != j:
                    dist = np.linalg.norm(pos[i] - pos[j])
                    if dist <= radius:
                        count += 1
            neighbor_counts.append(count)
    return np.mean(neighbor_counts)

agent_counts = [6, 9, 12]
radii = np.arange(0.1, 1.6, 0.05)

plt.figure(figsize=(10, 6))

for n in agent_counts:
    means = [avg_agent_neighbors(n, r) for r in radii]
    plt.plot(radii, means, label=f'N={n}', linewidth=2)
    # print table too
    print(f"\nN={n} agents:")
    print(f"{'Radius':>8} | {'Avg neighbors':>14}")
    print("-" * 26)
    for r, m in zip(radii, means):
        print(f"{r:>8.2f} | {m:>14.3f}")

plt.xlabel('Sensing radius ρ')
plt.ylabel('Avg neighbors per agent')
plt.title('Average agent neighbors vs sensing radius (world size = 2)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='1 neighbor')
plt.tight_layout()
plt.savefig('neighbor_analysis.png', dpi=150)
plt.show()
print("\nPlot saved to neighbor_analysis.png")