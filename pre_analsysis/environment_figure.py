import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

# ── LEFT PANEL ───────────────────────────────────────────────────────────────

ax1.set_xlim(-1.1, 1.1)
ax1.set_ylim(-1.1, 1.1)
ax1.set_aspect('equal')
ax1.set_facecolor('#f8f9fa')
ax1.set_title('(a) 2D Navigation Environment', fontsize=13, fontweight='bold', pad=10)
ax1.set_xlabel('x', fontsize=11)
ax1.set_ylabel('y', fontsize=11)
ax1.grid(True, alpha=0.2, zorder=0)

# World boundary
ax1.add_patch(patches.Rectangle((-1, -1), 2, 2, linewidth=2,
              edgecolor='#333333', facecolor='none', zorder=1))

RADIUS = 0.7
focal_pos = np.array([-0.15, 0.05])

# Sensing radius
ax1.add_patch(plt.Circle(focal_pos, RADIUS, color='#2196F3',
              fill=True, alpha=0.08, zorder=2))
ax1.add_patch(plt.Circle(focal_pos, RADIUS, color='#2196F3',
              fill=False, linewidth=1.8, linestyle='--', zorder=3))

# Radius label
ax1.annotate('', xy=(focal_pos[0] + RADIUS, focal_pos[1]),
             xytext=focal_pos,
             arrowprops=dict(arrowstyle='->', color='#2196F3', lw=1.2))
ax1.text(focal_pos[0] + RADIUS/2, focal_pos[1] + 0.07,
         r'$\rho$', color='#2196F3', fontsize=12, ha='center')

# Positions
agent_B = np.array([0.25, 0.40])   # inside radius
agent_D = np.array([0.75, -0.50])  # outside radius
obstacle = np.array([-0.40, -0.35]) # inside radius
goal_A = np.array([0.55, -0.20])   # focal agent's goal (outside radius)

# ── Edges ──
def offset_arrow(ax, p1, p2, color, lw, ls='-', r1=0.075, r2=0.075):
    dx = p2[0]-p1[0]; dy = p2[1]-p1[1]
    dist = np.sqrt(dx**2+dy**2)
    s = p1 + r1/dist * np.array([dx,dy])
    e = p2 - r2/dist * np.array([dx,dy])
    ax.annotate('', xy=e, xytext=s,
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, mutation_scale=12,
                                linestyle=ls))

# A <-> B bidirectional
offset_arrow(ax1, focal_pos, agent_B, '#1D9E75', 1.8, r2=0.07)
offset_arrow(ax1, agent_B, focal_pos, '#1D9E75', 1.8, r2=0.075)

# A -> Goal (unidirectional, dashed)
offset_arrow(ax1, focal_pos, goal_A, '#E24B4A', 1.5, ls='dashed', r2=0.05)

# Obstacle -> A (unidirectional)
offset_arrow(ax1, obstacle, focal_pos, '#888780', 1.3, r1=0.06, r2=0.075)

# ── Entities ──
# Obstacle
ax1.add_patch(plt.Circle(obstacle, 0.06, color='#aaaaaa',
              zorder=5, ec='#666666', lw=1))
ax1.text(obstacle[0], obstacle[1]-0.13, 'Obstacle',
         fontsize=8, ha='center', color='#555555')

# Goal A
ax1.add_patch(patches.Rectangle(
    (goal_A[0]-0.05, goal_A[1]-0.05), 0.10, 0.10,
    color='#E24B4A', zorder=5, ec='#a00000', lw=1))
ax1.text(goal_A[0], goal_A[1]-0.13, "Goal$_A$",
         fontsize=8, ha='center', color='#a00000')

# Agent B (inside)
ax1.add_patch(plt.Circle(agent_B, 0.075, color='#1D9E75',
              zorder=6, ec='white', lw=1.5))
ax1.text(agent_B[0], agent_B[1], 'B', color='white',
         fontsize=10, ha='center', va='center', fontweight='bold', zorder=7)

# Agent D (outside, faded)
ax1.add_patch(plt.Circle(agent_D, 0.075, color='#aaaaaa',
              zorder=6, ec='white', lw=1.5, alpha=0.5))
ax1.text(agent_D[0], agent_D[1], 'D', color='white',
         fontsize=10, ha='center', va='center', fontweight='bold', zorder=7, alpha=0.5)
ax1.text(agent_D[0], agent_D[1]-0.15, 'outside\nradius',
         fontsize=7.5, color='#aaaaaa', ha='center')

# Agent A (focal)
ax1.add_patch(plt.Circle(focal_pos, 0.075, color='#185FA5',
              zorder=6, ec='white', lw=2))
ax1.text(focal_pos[0], focal_pos[1], 'A', color='white',
         fontsize=10, ha='center', va='center', fontweight='bold', zorder=7)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#185FA5',
           markersize=10, label='Focal agent (A)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#1D9E75',
           markersize=10, label='Agent within radius'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#aaaaaa',
           markersize=10, alpha=0.5, label='Agent outside radius'),
    Line2D([0],[0], marker='s', color='w', markerfacecolor='#E24B4A',
           markersize=9, label='Goal$_A$'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#aaaaaa',
           markeredgecolor='#666666', markersize=10, label='Obstacle'),
    Line2D([0],[0], color='#1D9E75', lw=1.8, label='Agent↔Agent (bidirectional)'),
    Line2D([0],[0], color='#E24B4A', lw=1.5, linestyle='dashed', label='Agent→Goal (sensing)'),
    Line2D([0],[0], color='#888780', lw=1.3, label='Obstacle→Agent (sensing)'),
]
ax1.legend(handles=legend_elements, loc='upper left', fontsize=8, framealpha=0.9,
           bbox_to_anchor=(1.01, 1), borderaxespad=0)
# ── RIGHT PANEL: Graph ────────────────────────────────────────────────────────

ax2.set_xlim(0, 1)
ax2.set_ylim(-0.05, 1.05)
ax2.set_aspect('equal')
ax2.axis('off')
ax2.set_title(r"(b) Agent A's Local Graph $g^{(A)}$",
              fontsize=13, fontweight='bold', pad=10)

def draw_node(ax, x, y, line1, line2, color, r=0.09):
    ax.add_patch(plt.Circle((x, y), r, color=color, zorder=5, ec='white', lw=2))
    ax.text(x, y+0.025, line1, color='white', fontsize=9,
            ha='center', va='center', fontweight='bold', zorder=6)
    ax.text(x, y-0.025, line2, color='white', fontsize=7,
            ha='center', va='center', zorder=6)

def graph_arrow(ax, p1, p2, color, bidir=False, ls='-', r=0.09):
    dx = p2[0]-p1[0]; dy = p2[1]-p1[1]
    dist = np.sqrt(dx**2+dy**2)
    s = np.array(p1) + r/dist * np.array([dx,dy])
    e = np.array(p2) - r/dist * np.array([dx,dy])
    ax.annotate('', xy=e, xytext=s,
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=1.8, mutation_scale=13, linestyle=ls))
    if bidir:
        ax.annotate('', xy=s, xytext=e,
                    arrowprops=dict(arrowstyle='->', color=color,
                                    lw=1.8, mutation_scale=13, linestyle=ls))

# Node positions
nA  = (0.50, 0.62)
nB  = (0.50, 0.88)
nG  = (0.15, 0.62)
nO  = (0.85, 0.62)

# Arrows
graph_arrow(ax2, nB, nA, '#1D9E75', bidir=True)
graph_arrow(ax2, nG, nA, '#E24B4A', ls='dashed')
graph_arrow(ax2, nO, nA, '#888780')

# Edge labels
ax2.text(0.58, 0.76, 'bidirectional\n(comm)', fontsize=7,
         color='#1D9E75', ha='left', style='italic')
ax2.text(0.30, 0.67, 'unidirectional\n(sense)', fontsize=7,
         color='#E24B4A', ha='center', style='italic')
ax2.text(0.72, 0.67, 'unidirectional\n(sense)', fontsize=7,
         color='#888780', ha='center', style='italic')

# Neighbor nodes with labels inside
draw_node(ax2, *nB, 'Agent B', 'Δpos, Δvel, Δgoal', '#1D9E75')
draw_node(ax2, *nG, 'Goal', 'Δpos', '#E24B4A')
draw_node(ax2, *nO, 'Obstacle', 'Δpos', '#888888')

# Focal agent A
draw_node(ax2, *nA, 'Agent A', 'pos, vel, Δgoal', '#185FA5', r=0.10)

# Arrow from A down to GNN box
ax2.annotate('', xy=(0.50, 0.41), xytext=(0.50, 0.52),
             arrowprops=dict(arrowstyle='->', color='#534AB7',
                             lw=2.0, mutation_scale=14))

# GNN box — drawn as rect with text separately
ax2.add_patch(patches.Rectangle((0.18, 0.31), 0.64, 0.10,
              linewidth=1.5, edgecolor='#534AB7',
              facecolor='#eeeeff', zorder=4))
ax2.text(0.50, 0.36, r'GNN aggregation  →  $x^{(A)}_{agg}$',
         fontsize=9, ha='center', va='center',
         color='#534AB7', fontweight='bold', zorder=5)

# Arrow from GNN down to actor box
ax2.annotate('', xy=(0.50, 0.21), xytext=(0.50, 0.31),
             arrowprops=dict(arrowstyle='->', color='#333333',
                             lw=2.0, mutation_scale=14))

# Actor box
ax2.add_patch(patches.Rectangle((0.18, 0.11), 0.64, 0.10,
              linewidth=1.5, edgecolor='#1D9E75',
              facecolor='#eef5ee', zorder=4))
ax2.text(0.50, 0.16, r'Actor network  →  action $a^{(A)}$',
         fontsize=9, ha='center', va='center',
         color='#1D9E75', fontweight='bold', zorder=5)

# Agent D not included note
ax2.text(0.50, 0.99, 'Agent D: not in graph (outside ρ)',
         fontsize=8, ha='center', color='#aaaaaa', style='italic')









plt.suptitle('InforMARL Navigation Environment and Agent Graph Structure',
             fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('informarl_environment_figure.png', dpi=200, bbox_inches='tight')
plt.show()
print("Saved to informarl_environment_figure.png")