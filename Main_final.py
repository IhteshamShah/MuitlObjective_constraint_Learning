
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gridworld as gw
import MOCI_IRL as moci
import Sensitivity_Scalability_analysis as SSA





def define_mdp_and_demos():
    """Helper function to define the MDP and generate expert demonstrations."""
    # --- STEP 1: DEFINE GRIDWORLD SIZE ---

# 5×5 GridWorld            6×6 GridWorld              7×7 GridWorld                 8×8 GridWorld
# ------------------------------------------------------------------------------------------------
# [ 0  1  2  3  4 ]        [ 0  1  2  3  4  5 ]        [ 0  1  2  3  4  5  6 ]        [ 0  1  2  3  4  5  6  7 ]
# [ 5  6  7  8  9 ]        [ 6  7  8  9 10 11 ]        [ 7  8  9 10 11 12 13 ]        [ 8  9 10 11 12 13 14 15 ]
# [10 11 12 13 14 ]        [12 13 14 15 16 17 ]        [14 15 16 17 18 19 20 ]        [16 17 18 19 20 21 22 23 ]
# [15 16 17 18 19 ]        [18 19 20 21 22 23 ]        [21 22 23 24 25 26 27 ]        [24 25 26 27 28 29 30 31 ]
# [20 21 22 23 24 ]        [24 25 26 27 28 29 ]        [28 29 30 31 32 33 34 ]        [32 33 34 35 36 37 38 39 ]
#                          [30 31 32 33 34 35 ]        [35 36 37 38 39 40 41 ]        [40 41 42 43 44 45 46 47 ]
#                                                       [42 43 44 45 46 47 48 ]        [48 49 50 51 52 53 54 55 ]
#                                                                                      [56 57 58 59 60 61 62 63 ]
    '''
    GRID_SIZE = 5 

    # --- STEP 2: DEFINE TERRAIN STATES (indices) ---
    WATER = [12,13] # RIVER / HARD CONSTRAINTS
    GRASS = [3,7,14]
    ROCKS = [10,11,21]
    '''
    # --- STEP 2: DEFINE TERRAIN STATES (indices) ---
    GRID_SIZE = 8
    WATER = [12,17,38, 42, 43] # RIVER / HARD CONSTRAINTS
    GRASS = [3,7,12,13,29, 32, 33, 19,39,49]
    ROCKS = [20, 6,11,21,25,26,32,40,51,52,53]

    # --- STEP 3: DEFINE DEMONSTRATION COUNTS ---
    N_DEMOS_EXPERT1 = 10
    N_DEMOS_EXPERT2 = 10
    mdp = gw.CustomizableFeatureMDP(GRID_SIZE, WATER, GRASS, ROCKS)
    
    # Define Preferences [Sand, Grass, Rock, Water]
    w1 = np.array([1.0, 3.0, -1, -10.0]) # Expert 1: Grass Lover
    w2 = np.array([1.0, -1, 3.0, -10.0]) # Expert 2: Rock Lover
    
    # Generate Demos
    z1 = moci.backward_pass(mdp, w1, WATER)
    z2 = moci.backward_pass(mdp, w2, WATER)
    
    all_demos = [moci.sample_traj(mdp, w1, z1) for _ in range(N_DEMOS_EXPERT1)] + \
                [moci.sample_traj(mdp, w2, z2) for _ in range(N_DEMOS_EXPERT2)]
    
    # Mock responsibilities for visualization
    resp = np.zeros((N_DEMOS_EXPERT1 + N_DEMOS_EXPERT2, 2))
    resp[:N_DEMOS_EXPERT1, 0] = 1; resp[N_DEMOS_EXPERT2:, 1] = 1

    resp = np.zeros((N_DEMOS_EXPERT1 + N_DEMOS_EXPERT2, 2))
    resp[:N_DEMOS_EXPERT1, 0] = 1; resp[N_DEMOS_EXPERT2:, 1] = 1

    # Show Trajectories (Graph 2)
    gw.plot_grid_setup(mdp, "Expert Trajectories (Lime=Grass Preference, Orange=Rock Preference)", all_demos, resp)



    
    return w1, w2, WATER, mdp, all_demos, resp
# ==========================================
# EXECUTION SCRIPT
# ==========================================
if __name__ == "__main__":
    # Example: mdp = CustomizableFeatureMDP(GRID_SIZE, WATER, GRASS, ROCKS)
    # Example: all_demos = [...]
    w1,w2, WATER, mdp, all_demos, resp = define_mdp_and_demos()
    # Run the Expectation Maximization-MOCI (em_moci) framework
    
    inferred_c, final_weights, final_priors = moci.run_em_moci(mdp, all_demos, K=2, d_DKL=0.05, max_em_iters=10)


    print(f"Ground Truth WATER tiles: {WATER}")
    print(f"Algorithm Inferred Constraints: {list(inferred_c)}")

    # We use the same 'resp' array to keep the trajectory colors consistent.
    # Passing 'inferred_c' will trigger the red hatched boxes in your plotting function.
    title_inferred = "MOCI Inferred Constraints (Red Hatched)"



    gw.plot_grid_setup( mdp=mdp,  title=title_inferred, demos=all_demos, resp=resp, inf_c=inferred_c)  # <--- This replaces the ground-truth visualization with the algorithm's output


    print("Inferred Constraints:", sorted(list(inferred_c)))
    print("Final Weights:", final_weights)
    print("Final Priors:", final_priors)

    # Assuming final_weights[0] mapped to the Grass-Lover cluster
    print("Learned Preferences for Cluster 1:", np.round(final_weights[0], 2))
    # Expected output: Something like [ 0.1,  2.5, -1.8, -0.5]
    # High positive weight for index 1 (Grass), negative for index 2 (Rock)

    # Assuming final_weights[1] mapped to the Rock-Lover cluster
    print("Learned Preferences for Cluster 2:", np.round(final_weights[1], 2))
    # Expected output: Something like [-0.2, -2.1,  3.0, -0.4]
    # High positive weight for index 2 (Rock), negative for index 1 (Grass)

    gw.plot_preference_recovery(w1, w2, final_weights, features=['Sand', 'Grass', 'Rocks', 'Water'])

    SSA.run_sensitivity_and_scalability_experiments()



