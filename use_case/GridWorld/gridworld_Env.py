"""GridWorld environment and plotting utilities for the MOCI-IRL use case. Defines
CustomizableFeatureMDP, a grid MDP with sand/grass/rock/water terrain, deterministic
transitions, and a per-state feature map. Provides matplotlib helpers to draw a single
grid panel (terrain, start/goal callouts, inferred-constraint hatching, expert
trajectories) with a bold side legend, a single-figure and a two-panel (a)/(b) plot
function, and a grouped bar chart comparing ground-truth to learned preference
weights."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

RESULTS_DIR = Path(__file__).resolve().parents[2] / "Results" / "results_gridworld"


class CustomizableFeatureMDP:
    def __init__(self, size, water_states, grass_states, rock_states, horizon=30):
        """Args: size, water_states, grass_states, rock_states, horizon.
        Builds the grid's transitions and per-state feature map; returns None."""
        self.size = size
        self.num_states = size * size
        self.num_actions = 5
        self.horizon = horizon
        self.start_state = 0
        self.goal_state = self.num_states - 1

        self.num_features = 4
        self.feature_grid = np.zeros(self.num_states, dtype=int)

        for s in grass_states: self.feature_grid[s] = 1
        for s in rock_states: self.feature_grid[s] = 2
        for s in water_states: self.feature_grid[s] = 3

        self.water_states = water_states

        self.transitions = np.zeros((self.num_states, self.num_actions), dtype=int)
        for r in range(size):
            for c in range(size):
                s = r * size + c
                self.transitions[s, 0] = max(0, r-1) * size + c
                self.transitions[s, 1] = min(size-1, r+1) * size + c
                self.transitions[s, 2] = r * size + max(0, c-1)
                self.transitions[s, 3] = r * size + min(size-1, c+1)
                self.transitions[s, 4] = s

        self.feature_map = np.zeros((self.num_states, self.num_actions, self.num_features))
        for s in range(self.num_states):
            f_idx = self.feature_grid[s]
            self.feature_map[s, :, f_idx] = 1.0


def _draw_grid_panel(ax, mdp, demos=None, resp=None, inf_c=None, show_terrain_legend=True, show_trajectories=True):
    """Args: ax, mdp, demos, resp, inf_c, show_terrain_legend, show_trajectories.
    Draws terrain, start/goal callouts, inferred constraints, and demo trajectories onto ax.
    Returns: list of legend handles."""
    import matplotlib.lines as mlines

    ax.set_aspect('equal')

    feat_colors = {0: 'wheat', 1: 'forestgreen', 2: 'saddlebrown', 3: 'royalblue'}
    feat_labels = {0: 'Normal Tiles', 1: 'Grass Tiles', 2: 'Rock Tiles', 3: 'Water (Hard Constraint)'}

    for s in range(mdp.num_states):
        r, c = s // mdp.size, s % mdp.size
        color = feat_colors[mdp.feature_grid[s]]
        ax.add_patch(patches.Rectangle((c-0.5, mdp.size-1-r-0.5), 1, 1, color=color, alpha=0.3))

    ax.set_xlim(-0.5, mdp.size - 0.5)
    ax.set_ylim(-0.5, mdp.size - 0.5)

    def _cell_center(s):
        """Args: s (state index). Returns: (x, y) plot coordinates of the state's cell center."""
        r, c = s // mdp.size, s % mdp.size
        return c, mdp.size - 1 - r

    start_x, start_y = _cell_center(mdp.start_state)
    ax.text(
        start_x, start_y, 'Start', ha='center', va='center', fontsize=12,
        fontweight='bold', color='white', zorder=5,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='gold', edgecolor='#b8860b', linewidth=1.5),
    )

    goal_x, goal_y = _cell_center(mdp.goal_state)
    ax.text(
        goal_x, goal_y, 'Goal', ha='center', va='center', fontsize=12,
        fontweight='bold', color='white', zorder=5,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='tomato', edgecolor='#a52a2a', linewidth=1.5),
    )

    legend_handles = []
    if show_terrain_legend:
        present_features = set(int(f) for f in mdp.feature_grid)
        legend_handles = [
            patches.Patch(facecolor=feat_colors[f], alpha=0.5, edgecolor='none', label=feat_labels[f])
            for f in sorted(feat_colors) if f in present_features
        ]

    if inf_c:
        for s in inf_c:
            r, c = s // mdp.size, s % mdp.size
            ax.add_patch(patches.Rectangle((c-0.5, mdp.size-1-r-0.5), 1, 1, fill=False, hatch='///', edgecolor='red', lw=2))
        legend_handles.append(
            patches.Patch(facecolor='white', hatch='///', edgecolor='red', lw=2, label='Inferred Constraints')
        )

    if demos is not None and show_trajectories:
        line_colors = ['lime', 'orange']
        line_labels = ['Expert 1 (Grass-Lover)', 'Expert 2 (Rock-Lover)']
        seen_clusters = set()
        for i, d in enumerate(demos):
            c_id = np.argmax(resp[i])
            coords = np.array([(s % mdp.size, mdp.size - 1 - (s // mdp.size)) for s in d])
            ax.plot(coords[:, 0], coords[:, 1], color=line_colors[c_id], alpha=0.8, linewidth=3)
            seen_clusters.add(c_id)
        legend_handles.extend(
            mlines.Line2D([], [], color=line_colors[c_id], linewidth=3, label=line_labels[c_id])
            for c_id in sorted(seen_clusters)
        )

    return legend_handles


def _place_side_legend(ax, legend_handles, max_per_col=4):
    """Args: ax, legend_handles, max_per_col.
    Draws a bold legend to the right of ax, columns ordered bottom-to-top; returns None."""
    import math

    n = len(legend_handles)
    if n == 0:
        return
    ncol = max(1, math.ceil(n / max_per_col))
    nrows = math.ceil(n / ncol)
    columns = [legend_handles[i * nrows:(i + 1) * nrows] for i in range(ncol)]
    reordered = [h for col in columns for h in reversed(col)]

    ax.legend(
        handles=reordered, loc='center left', bbox_to_anchor=(1.02, 0.5),
        frameon=False, fontsize=11, borderaxespad=0.0, ncol=ncol,
        prop={'weight': 'bold'},
    )


def plot_grid_setup(mdp, title, demos=None, resp=None, inf_c=None, show_terrain_legend=True, show_trajectories=True):
    """Args: mdp, title, demos, resp, inf_c, show_terrain_legend, show_trajectories.
    Draws one grid panel and saves RESULTS_DIR/<title>.png; returns None.
    Note: title is set after savefig intentionally; do not reorder."""
    fig, ax = plt.subplots(figsize=(8, 8))
    legend_handles = _draw_grid_panel(ax, mdp, demos, resp, inf_c, show_terrain_legend, show_trajectories)
    _place_side_legend(ax, legend_handles)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(RESULTS_DIR / f'{title}.png', dpi=600, bbox_inches='tight')
    plt.title(title)
    plt.show()


def plot_grid_subfigures(mdp, demos, resp, inf_c, title_a='(a) Expert Trajectories',
                         title_b='(b) Inferred Constraints',
                         out_name='Expert_Trajectories_and_MOCI_Inferred_Constraints'):
    """Args: mdp, demos, resp, inf_c, title_a, title_b, out_name.
    Draws two panels ((a) trajectories, (b) inferred constraints) with side legends and
    saves RESULTS_DIR/<out_name>.png; returns None."""
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(15, 8))

    handles_a = _draw_grid_panel(ax_a, mdp, demos=demos, resp=resp, inf_c=None,
                                 show_terrain_legend=True, show_trajectories=True)
    _place_side_legend(ax_a, handles_a, max_per_col=len(handles_a))
    ax_a.set_title(title_a, fontsize=13, fontweight='bold', y=-0.14)

    handles_b = _draw_grid_panel(ax_b, mdp, demos=None, resp=None, inf_c=inf_c,
                                 show_terrain_legend=False, show_trajectories=False)
    _place_side_legend(ax_b, handles_b)
    ax_b.set_title(title_b, fontsize=13, fontweight='bold', y=-0.14)

    plt.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(RESULTS_DIR / f'{out_name}.png', dpi=600, bbox_inches='tight')
    plt.show()



def plot_preference_recovery(w1_true, w2_true, w_learned, features=['Sand', 'Grass', 'Rocks', 'Water']):
    """Args: w1_true, w2_true, w_learned, features.
    Draws grouped bar charts of normalized ground-truth vs learned weights per expert and
    saves RESULTS_DIR/preference_recovery_barchart.png; returns None."""
    def normalize(w):
        """Args: w (weight vector). Returns: w scaled to unit norm."""
        return w / (np.linalg.norm(w) + 1e-8)

    if w_learned[0][1] > w_learned[0][2]:
        print("Mapping Cluster 0 to Expert 1 (Grass-Lover)")
        w1_learned = w_learned[0]
        w2_learned = w_learned[1]
    else:
        print("Mapping Cluster 0 to Expert 2 (Rock-Lover)")
        w1_learned = w_learned[1]
        w2_learned = w_learned[0]

    gt_1 = normalize(w1_true)
    gt_2 = normalize(w2_true)
    lrn_1 = normalize(w1_learned)
    lrn_2 = normalize(w2_learned)

    x = np.arange(len(features))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(x - width/2, gt_1, width, label='Ground Truth', color='lightgray', edgecolor='black')
    ax1.bar(x + width/2, lrn_1, width, label='MOCI Learned', color='forestgreen', edgecolor='black')
    ax1.set_title('Expert 1 (Grass-Lover) Preferences', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(features, fontsize=12)
    ax1.axhline(0, color='black', linewidth=1)
    ax1.legend()

    ax2.bar(x - width/2, gt_2, width, label='Ground Truth', color='lightgray', edgecolor='black')
    ax2.bar(x + width/2, lrn_2, width, label='MOCI Learned', color='saddlebrown', edgecolor='black')
    ax2.set_title('Expert 2 (Rock-Lover) Preferences', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(features, fontsize=12)
    ax2.axhline(0, color='black', linewidth=1)
    ax2.legend()

    plt.suptitle('Joint Recovery of Heterogeneous Preferences', fontsize=16, fontweight='bold')
    plt.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(RESULTS_DIR / 'preference_recovery_barchart.png', dpi=600, bbox_inches='tight')
    plt.show()
