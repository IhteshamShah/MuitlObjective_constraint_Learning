import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gridworld as gw
from MOCL_IRL import backward_pass, sample_traj, get_log_likelihood



# # ==========================================
# # 4. Main Execution
# # ==========================================
# if __name__ == "__main__":
#     mdp = FeatureGridMDP()
    
#     # Show Setup (Graph 1)
#     plot_grid_setup(mdp, "Graph 1: Environment Setup (Blue=Water, Green=Grass, Brown=Rock)")

#     # Expert Preferences: [Sand, Grass, Rock, Water]
#     # Expert 1 likes Grass (+2), Expert 2 likes Rock (+2)
#     w1 = np.array([0.0, 2.0, 0.0, -5.0]) 
#     w2 = np.array([0.0, 0.0, 2.0, -5.0])
    
#     # True constraints are water
#     z1 = backward_pass(mdp, w1, mdp.water_states)
#     z2 = backward_pass(mdp, w2, mdp.water_states)
    
#     demos = [sample_traj(mdp, w1, z1) for _ in range(5)] + [sample_traj(mdp, w2, z2) for _ in range(5)]
#     # Mock responsibilities for visualization
#     resp = np.zeros((10, 2))
#     resp[:5, 0] = 1; resp[5:, 1] = 1

#     # Show Trajectories (Graph 2)
#     plot_grid_setup(mdp, "Graph 2: Expert Trajectories (Lime=Grass Preference, Orange=Rock Preference)", demos, resp)

#     # Simplified Constraint Inference
#     inferred_c = []
#     candidates = [s for s in range(mdp.num_states) if not any(s in d for d in demos) and s != mdp.goal_state]
    
#     print("Inferring constraints...")
#     for _ in range(4): # Run 4 steps of MLCI
#         best_cand, best_score = None, -np.inf
#         for cand in np.random.choice(candidates, 15):
#             test_c = inferred_c + [cand]
#             z_test1 = backward_pass(mdp, w1, test_c)
#             z_test2 = backward_pass(mdp, w2, test_c)
#             score = sum(get_log_likelihood(d, mdp, w1, z_test1) for d in demos[:5]) + \
#                     sum(get_log_likelihood(d, mdp, w2, z_test2) for d in demos[5:])
#             if score > best_score:
#                 best_score, best_cand = score, cand
#         if best_cand:
#             inferred_c.append(best_cand)
#             candidates.remove(best_cand)

#     # Show Final Constraints (Graph 3)
#     plot_grid_setup(mdp, "Graph 3: True vs Inferred Constraints (Hatched Red = Inferred)", None, None, inferred_c)









# ==========================================
# 4. CONFIGURATION & EXECUTION
# ==========================================
# [ 0  1  2  3  4  5  6  7 ]
# [ 8  9 10 11 12 13 14 15 ]
# [16 17 18 19 20 21 22 23 ]
# [24 25 26 27 28 29 30 31 ]
# [32 33 34 35 36 37 38 39 ]
# [40 41 42 43 44 45 46 47 ]
# [48 49 50 51 52 53 54 55 ]
# [56 57 58 59 60 61 62 63 ]

# --- STEP 1: DEFINE GRIDWORLD SIZE ---
GRID_SIZE = 7 

# --- STEP 2: DEFINE TERRAIN STATES (indices) ---
WATER = [15, 16, 17, 18, 24, 31] # RIVER / HARD CONSTRAINTS
GRASS = [2, 3, 9, 10, 22, 23, 29, 30]
ROCKS = [4, 5, 11, 12, 25, 26, 32, 33, 39, 40]

# --- STEP 3: DEFINE DEMONSTRATION COUNTS ---
N_DEMOS_EXPERT1 = 10
N_DEMOS_EXPERT2 = 10

# --- RUN SIMULATION ---
if __name__ == "__main__":
    # Create MDP
    mdp = gw.CustomizableFeatureMDP(GRID_SIZE, WATER, GRASS, ROCKS)
    
    # Define Preferences [Sand, Grass, Rock, Water]
    w1 = np.array([0.0, 3.0, 0.0, -10.0]) # Expert 1: Grass Lover
    w2 = np.array([0.0, 0.0, 3.0, -10.0]) # Expert 2: Rock Lover
    
    # Generate Demos
    z1 = backward_pass(mdp, w1, WATER)
    z2 = backward_pass(mdp, w2, WATER)
    
    all_demos = [sample_traj(mdp, w1, z1) for _ in range(N_DEMOS_EXPERT1)] + \
                [sample_traj(mdp, w2, z2) for _ in range(N_DEMOS_EXPERT2)]
    # Mock responsibilities for visualization
    resp = np.zeros((N_DEMOS_EXPERT1 + N_DEMOS_EXPERT2, 2))
    resp[:N_DEMOS_EXPERT1, 0] = 1; resp[N_DEMOS_EXPERT2:, 1] = 1

    # Show Trajectories (Graph 2)
    gw.plot_grid_setup(mdp, "Graph 2: Expert Trajectories (Lime=Grass Preference, Orange=Rock Preference)", all_demos, resp)