
import time 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gridworld as gw
import MOCI_IRL as moci
from max_Likly_CI_ExistingWork import run_algo


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

GRID_SIZE = 6 

# --- STEP 2: DEFINE TERRAIN STATES (indices) ---
WATER = [12,13,22,26] # RIVER / HARD CONSTRAINTS
GRASS = [3,7,14]
ROCKS = [10,11,21]

# # --- STEP 2: DEFINE TERRAIN STATES (indices) ---
# GRID_SIZE = 5
# WATER = [12,17,38, 42, 43] # RIVER / HARD CONSTRAINTS
# GRASS = [3,7,12,13,29, 32, 33, 19,39,49]
# ROCKS = [20, 6,11,21,25,26,32,40,51,52,53]

# --- STEP 3: DEFINE DEMONSTRATION COUNTS ---
N_DEMOS_EXPERT1 = 10
N_DEMOS_EXPERT2 = 10
mdp = gw.CustomizableFeatureMDP(GRID_SIZE, WATER, GRASS, ROCKS)

# Define Preferences [Sand, Grass, Rock, Water]
w1 = np.array([1.0, 3.0, -1, -10.0]) # Expert 1: Grass Lover
w2 = np.array([1.0, -1, 3.0, -10.0]) # Expert 2: Rock Lover

def sample_traj_both_formats(mdp, weights, Z):
    """
    Generates a single expert demonstration and returns it in two formats.
    
    Returns:
        traj_states (list): List of states e.g., [0, 1, 2, 3, 24]
        traj_sa (list): List of (state, action) tuples e.g., [(0, 3), (1, 3), (2, 3), (3, 1), (24, None)]
    """
    curr = mdp.start_state
    
    traj_states = [curr]
    traj_sa = []
    
    rew = np.dot(mdp.feature_map, weights)
    
    for t in range(mdp.horizon - 1):
        if curr == mdp.goal_state: 
            break
            
        p = [np.exp(rew[curr, a]) * Z[mdp.transitions[curr, a], t+1] for a in range(5)]
        if sum(p) == 0: 
            break
            
        # Sample the action based on probabilities
        chosen_action = np.random.choice(5, p=np.array(p)/sum(p))
        
        # Record the State-Action pair for Format 2
        traj_sa.append((curr, chosen_action))
        
        # Transition to next state
        curr = mdp.transitions[curr, chosen_action]
        
        # Record the new state for Format 1
        traj_states.append(curr)
        
    # Append the final state with 'None' for Format 2 (end of trajectory)
    traj_sa.append((curr, None))
    
    return traj_states, traj_sa



def calculate_cmse(true_constraints, inferred_constraints, total_states):
    """
    Calculates the Constraint Mean Squared Error (CMSE) for state-only constraints.
    
    Args:
        true_constraints (set or list): Indices of the true constrained states.
        inferred_constraints (set or list): Indices of the states inferred as constraints.
        total_states (int): The total number of states in the uniformly discretized space.
        
    Returns:
        float: The CMSE value.
    """
    # Create binary arrays representing the constraint functions
    true_array = np.zeros(total_states)
    inferred_array = np.zeros(total_states)
    
    # Assign 1 to constrained states
    for s in true_constraints:
        true_array[s] = 1.0
        
    for s in inferred_constraints:
        inferred_array[s] = 1.0
        
    # Calculate Mean Squared Error
    # Since values are 0 or 1, (true - inferred)^2 is 1 for mismatches, 0 for matches
    mse = np.mean((true_array - inferred_array) ** 2)
    
    return mse

def define_mdp_and_demos():

    
    # Generate Demos
    z1 = moci.backward_pass(mdp, w1, WATER)
    z2 = moci.backward_pass(mdp, w2, WATER)
    
    # Initialize the two dataset lists
    demos_moci_format = []  # List of state-only lists 
    demos_mlci_format = []  # List of state-action tuple lists

    # Generate for Expert 1
    for _ in range(N_DEMOS_EXPERT1):
        t_states, t_sa = sample_traj_both_formats(mdp, w1, z1)
        demos_moci_format.append(t_states)
        demos_mlci_format.append(t_sa)

    # Generate for Expert 2
    for _ in range(N_DEMOS_EXPERT2):
        t_states, t_sa = sample_traj_both_formats(mdp, w2, z2)
        demos_moci_format.append(t_states)
        demos_mlci_format.append(t_sa)

    # # --- Verification ---
    # print("--- MOCI Format (States Only) ---")
    # print(demos_moci_format[0])

    # print("\n--- MLCI Format (State-Action Pairs) ---")
    # print(demos_mlci_format[0])
    
    # Mock responsibilities for visualization
    resp = np.zeros((N_DEMOS_EXPERT1 + N_DEMOS_EXPERT2, 2))
    resp[:N_DEMOS_EXPERT1, 0] = 1; resp[N_DEMOS_EXPERT2:, 1] = 1

    resp = np.zeros((N_DEMOS_EXPERT1 + N_DEMOS_EXPERT2, 2))
    resp[:N_DEMOS_EXPERT1, 0] = 1; resp[N_DEMOS_EXPERT2:, 1] = 1

    # Show Trajectories (Graph 2)
    gw.plot_grid_setup(mdp, "Expert Trajectories (Lime=Grass Preference, Orange=Rock Preference)", demos_moci_format, resp)



    
    return GRID_SIZE,w1, w2, WATER, mdp, demos_mlci_format, demos_moci_format, resp
# ==========================================
# EXECUTION SCRIPT
# ==========================================
if __name__ == "__main__":
    # Example: mdp = CustomizableFeatureMDP(GRID_SIZE, WATER, GRASS, ROCKS)
    # Example: all_demos = [...]
    GRID_SIZE, w1,w2, WATER, mdp, demos_mlci_format, demos_moci_format, resp = define_mdp_and_demos()
    # Run the Expectation Maximization-MOCI (em_moci) framework
    
    # --- Calculate Time and constraint for MOCI ---
    start_moci = time.time()
    inferred_c, final_weights, final_priors = moci.run_em_moci( mdp, demos_moci_format, K=2, d_DKL=0.05, max_em_iters=1)
    end_moci = time.time()
    moci_duration = end_moci - start_moci

    # --- Calculate Time and constraints for MLCI ---
    start_mlci = time.time()
    inferred_constraints = run_algo(demos_mlci_format, WATER, GRID_SIZE)
    end_mlci = time.time()
    mlci_duration = end_mlci - start_mlci

    # --- Calculate mean squared errors for MOCI and MLCI ---
    mse_moci = calculate_cmse(WATER, inferred_c, mdp.num_states)
    mse_mlci = calculate_cmse(WATER, inferred_constraints, mdp.num_states)

    
    
    # --- Print the results ---
    
    print("\nFinal Result:")
    print(f"Ground Truth WATER tiles: {WATER}",f"Inferred Constraints MOCI: {list(inferred_c)}", f"Inferred Constraints MLCI: {inferred_constraints}")

    print(f"Constraint Mean Squared Error for MOCI: {mse_moci}")
    print(f"Constraint Mean Squared Error for MLCI: {mse_mlci}")

    print(f"MOCI Run-time: {moci_duration:.4f} seconds")
    print(f"MLCI Run-time: {mlci_duration:.4f} seconds")
