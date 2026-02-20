import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# ==========================================
# 2. Logic Kernels (MaxEnt IRL)
# ==========================================
def backward_pass(mdp, weights, obstacles):
    Z = np.zeros((mdp.num_states, mdp.horizon + 1))
    rewards = np.dot(mdp.feature_map, weights)
    Z[:, mdp.horizon] = 1.0
    Z[mdp.goal_state, mdp.horizon] = np.exp(10.0) 
    for obs in obstacles: Z[obs, :] = 0.0

    for t in range(mdp.horizon - 1, -1, -1):
        for s in range(mdp.num_states):
            if s in obstacles: continue
            if s == mdp.goal_state:
                Z[s, t] = np.exp(10.0) * Z[s, t+1]
                continue
            z_sum = 0.0
            for a in range(mdp.num_actions):
                sn = mdp.transitions[s, a]
                if Z[sn, t+1] > 0:
                    z_sum += np.exp(rewards[s, a]) * (Z[sn, t+1] ** 0.9)
            Z[s, t] = z_sum
    return Z

def sample_traj(mdp, weights, Z):
    curr, traj = mdp.start_state, [mdp.start_state]
    rew = np.dot(mdp.feature_map, weights)
    for t in range(mdp.horizon - 1):
        if curr == mdp.goal_state: break
        p = [np.exp(rew[curr, a]) * (Z[mdp.transitions[curr, a], t+1]**0.9) for a in range(5)]
        if sum(p) == 0: break
        curr = mdp.transitions[curr, np.random.choice(5, p=np.array(p)/sum(p))]
        traj.append(curr)
    return traj

def get_log_likelihood(demo, mdp, weights, Z):
    if Z[mdp.start_state, 0] <= 0: return -1e9
    rewards = np.dot(mdp.feature_map, weights)
    path_reward = sum(rewards[demo[i], 0] for i in range(len(demo)-1))
    if demo[-1] == mdp.goal_state: path_reward += 10.0
    return path_reward - np.log(Z[mdp.start_state, 0])

# ==========================================
# 3. Visualization
# ==========================================
def plot_grid(mdp, title, demos=None, inferred_c=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    feat_colors = {0: 'wheat', 1: 'forestgreen', 2: 'saddlebrown', 3: 'royalblue'}
    
    for s in range(mdp.num_states):
        r, c = s // mdp.size, s % mdp.size
        ax.add_patch(patches.Rectangle((c-0.5, mdp.size-1-r-0.5), 1, 1, color=feat_colors[mdp.feature_grid[s]], alpha=0.4))
    
    if demos:
        # First half are Expert 1, second half Expert 2
        half = len(demos) // 2
        for i, d in enumerate(demos):
            color = 'lime' if i < half else 'orange'
            coords = np.array([(s % mdp.size, mdp.size - 1 - (s // mdp.size)) for s in d])
            ax.plot(coords[:, 0], coords[:, 1], color=color, alpha=0.7, linewidth=2)

    if inferred_c:
        for s in inferred_c:
            r, c = s // mdp.size, s % mdp.size
            ax.add_patch(patches.Rectangle((c-0.5, mdp.size-1-r-0.5), 1, 1, fill=False, hatch='XX', edgecolor='red', lw=2))

    plt.title(title)
    plt.show()

