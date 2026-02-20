
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

    plt.title(title)
    plt.show()