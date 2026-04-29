
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==========================================
# 1.  Multi-Feature GridMDP
# ==========================================
class CustomizableFeatureMDP:
    def __init__(self, size, water_states, grass_states, rock_states, horizon=30):
        self.size = size
        self.num_states = size * size
        self.num_actions = 5  # U, D, L, R, Stay
        self.horizon = horizon
        self.start_state = 0
        self.goal_state = self.num_states - 1
        
        # Feature Indices: 0:Sand (Default), 1:Grass, 2:Rock, 3:Water
        self.num_features = 4
        self.feature_grid = np.zeros(self.num_states, dtype=int)
        
        # Assign terrain based on input lists
        for s in grass_states: self.feature_grid[s] = 1
        for s in rock_states: self.feature_grid[s] = 2
        for s in water_states: self.feature_grid[s] = 3
        
        # Store for reference
        self.water_states = water_states
        
        # Build Transitions
        self.transitions = np.zeros((self.num_states, self.num_actions), dtype=int)
        for r in range(size):
            for c in range(size):
                s = r * size + c
                self.transitions[s, 0] = max(0, r-1) * size + c # Up
                self.transitions[s, 1] = min(size-1, r+1) * size + c # Down
                self.transitions[s, 2] = r * size + max(0, c-1) # Left
                self.transitions[s, 3] = r * size + min(size-1, c+1) # Right
                self.transitions[s, 4] = s # Stay

        # Build Feature Map (S, A, F)
        self.feature_map = np.zeros((self.num_states, self.num_actions, self.num_features))
        for s in range(self.num_states):
            f_idx = self.feature_grid[s]
            self.feature_map[s, :, f_idx] = 1.0
# ==========================================
# 3. Visualization Logic
# ==========================================
def plot_grid_setup(mdp, title, demos=None, resp=None, inf_c=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    
    # Colors: 0:Sand(Wheat), 1:Grass(Green), 2:Rock(Brown), 3:Water(Blue)
    feat_colors = {0: 'wheat', 1: 'forestgreen', 2: 'saddlebrown', 3: 'royalblue'}
    
    # Draw Terrain
    for s in range(mdp.num_states):
        r, c = s // mdp.size, s % mdp.size
        color = feat_colors[mdp.feature_grid[s]]
        ax.add_patch(patches.Rectangle((c-0.5, mdp.size-1-r-0.5), 1, 1, color=color, alpha=0.3))
    
    # Draw Inferred Constraints
    if inf_c:
        for s in inf_c:
            r, c = s // mdp.size, s % mdp.size
            ax.add_patch(patches.Rectangle((c-0.5, mdp.size-1-r-0.5), 1, 1, fill=False, hatch='///', edgecolor='red', lw=2))

    # Draw Demos
    if demos is not None:
        line_colors = ['lime', 'orange'] # Expert 1 (Grass-lover), Expert 2 (Rock-lover)
        for i, d in enumerate(demos):
            c_id = np.argmax(resp[i])
            coords = np.array([(s % mdp.size, mdp.size - 1 - (s // mdp.size)) for s in d])
            ax.plot(coords[:, 0], coords[:, 1], color=line_colors[c_id], alpha=0.8, linewidth=3)

    plt.savefig(f'Results/{title}.png', dpi=600, bbox_inches='tight')
    plt.title(title)
    plt.show()



def plot_preference_recovery(w1_true, w2_true, w_learned, features=['Sand', 'Grass', 'Rocks', 'Water']):
    """
    Plots a grouped bar chart comparing normalized ground-truth weights 
    to the learned weights from the MOCI algorithm.
    """
    # Helper function to normalize weights for fair visual comparison
    def normalize(w):
        return w / (np.linalg.norm(w) + 1e-8)
    
    # --- UPDATED MATCHING LOGIC ---
    # Inspect Cluster 1 (w_learned[0])
    # Index 1 is Grass, Index 2 is Rocks
    if w_learned[0][1] > w_learned[0][2]:
        print("Mapping Cluster 0 to Expert 1 (Grass-Lover)")
        w1_learned = w_learned[0]
        w2_learned = w_learned[1]
    else:
        print("Mapping Cluster 0 to Expert 2 (Rock-Lover)")
        w1_learned = w_learned[1]
        w2_learned = w_learned[0]

    # Normalize all weights so the scales match visually
    gt_1 = normalize(w1_true)
    gt_2 = normalize(w2_true)
    lrn_1 = normalize(w1_learned)
    lrn_2 = normalize(w2_learned)

    x = np.arange(len(features))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Expert 1 (Grass Lover)
    ax1.bar(x - width/2, gt_1, width, label='Ground Truth', color='lightgray', edgecolor='black')
    ax1.bar(x + width/2, lrn_1, width, label='MOCI Learned', color='forestgreen', edgecolor='black')
    ax1.set_title('Expert 1 (Grass-Lover) Preferences', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(features, fontsize=12)
    ax1.axhline(0, color='black', linewidth=1)
    ax1.legend()

    # Plot Expert 2 (Rock Lover)
    ax2.bar(x - width/2, gt_2, width, label='Ground Truth', color='lightgray', edgecolor='black')
    ax2.bar(x + width/2, lrn_2, width, label='MOCI Learned', color='saddlebrown', edgecolor='black')
    ax2.set_title('Expert 2 (Rock-Lover) Preferences', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(features, fontsize=12)
    ax2.axhline(0, color='black', linewidth=1)
    ax2.legend()

    plt.suptitle('Joint Recovery of Heterogeneous Preferences', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('Results/preference_recovery_barchart.png', dpi=600, bbox_inches='tight')
    plt.show()
