import numpy as np
from scipy.special import logsumexp
import gridworld as gw

"""
Overview of the main components of the EM-based MLCI framework:

1. calculate_trajectory_prob(mdp, xi, C, w_k)
   Inputs:
     - mdp: The MDP environment
     - xi: A single demonstrated trajectory
     - C: Current set of inferred constraints
     - w_k: Reward weights for expert cluster k

   Output:
     - P(xi | C, w_k): Probability that cluster k generated trajectory xi

   Description:
     Implements the Maximum Entropy (MaxEnt) trajectory distribution.
     If the trajectory violates any constraint in C, the probability is 0
     (via the indicator function I^C(xi)). Otherwise, it computes the
     partition function Z(C, w_k) using a backward pass, evaluates the
     trajectory reward R_{w_k}(xi), and returns exp(R) / Z.


2. calculate_joint_log_likelihood(mdp, D, C, weights, priors)
   Inputs:
     - mdp: The MDP environment
     - D: Set of all demonstrated trajectories
     - C: Current constraints
     - weights: Reward weights for all clusters
     - priors: Prior probability for each cluster

   Output:
     - L: Total joint log-likelihood of the dataset

   Description:
     Computes the marginal log-likelihood of the demonstrations by
     summing over latent expert clusters. For each trajectory, the
     probability under each cluster is weighted by its prior and summed
     before taking the logarithm.


3. identify_candidates(mdp, D)
   Inputs:
     - mdp: The MDP environment
     - D: Set of demonstrations

   Output:
     - candidates: List of states that are candidate constraints

   Description:
     A state is considered a candidate constraint if it is never visited
     by any expert. States visited by experts cannot be hard constraints.


4. e_step(mdp, D, C_hat, weights, priors)
   Inputs:
     - mdp: The MDP environment
     - D: Demonstrations
     - C_hat: Current inferred constraints
     - weights: Cluster reward weights
     - priors: Cluster priors

   Output:
     - gamma: Responsibility matrix of shape (num_demos, K)

   Description:
     Expectation step of EM. Computes the posterior probability that each
     cluster k generated demonstration i (gamma_{i,k}).


5. m_step_weights(mdp, D, C_hat, weights, responsibilities, lr, steps)
   Inputs:
     - mdp: The MDP environment
     - D: Demonstrations
     - C_hat: Current constraints
     - weights: Current reward weights
     - responsibilities: Posterior responsibilities gamma
     - lr: Learning rate
     - steps: Number of gradient steps

   Output:
     - Updated reward weights for each cluster

   Description:
     Performs MaxEnt Inverse Reinforcement Learning (IRL). Updates reward
     weights by matching empirical and expected feature counts. Each
     cluster’s gradient is weighted by its responsibility, ensuring that
     clusters adapt primarily to trajectories they explain.


6. m_step_constraints(mdp, D, C_hat, weights, priors, d_DKL)
   Inputs:
     - mdp: The MDP environment
     - D: Demonstrations
     - C_hat: Current inferred constraints
     - weights: Updated reward weights
     - priors: Updated cluster priors
     - d_DKL: KL-divergence stopping threshold

   Output:
     - Updated constraint set C_hat

   Description:
     Core step of the MLCI algorithm. Iteratively tests adding candidate
     constraints and selects the one that maximally increases the joint
     log-likelihood. Stops when the improvement (equivalent to a decrease
     in KL-divergence) falls below d_DKL.


7. run_em_mlci(mdp, D, K, d_DKL, max_em_iters)
   Inputs:
     - mdp: The MDP environment
     - D: Demonstrations
     - K: Number of expert clusters
     - d_DKL: KL-divergence stopping threshold
     - max_em_iters: Maximum EM iterations

   Output:
     - C_hat: Final inferred shared constraints
     - weights: Learned reward weights
     - priors: Learned cluster priors

   Description:
     Main orchestration function. Initializes parameters and alternates
     between the E-step and M-steps (updating priors, reward weights, and
     constraints) until convergence or the maximum number of iterations
     is reached.
"""

# ==========================================
# HELPER: MaxEnt Core (Z and Probabilities)
# ==========================================
def backward_pass(mdp, weights, obstacles):
    """Computes Partition Function Z(C, w_k) in log-space to prevent underflow."""
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
                    z_sum += np.exp(rewards[s, a]) * Z[sn, t+1]
            Z[s, t] = z_sum
    return Z

def calculate_trajectory_prob(mdp, xi, C, w_k, Z_matrix):
    """
    Math: P(xi | C, w_k) = (e^{R_{w_k}(xi)} / Z(C, w_k)) * I^C(xi)
    Returns the log probability for stability. Returns -inf if impossible.
    """
    # Indicator function I^C(xi)
    for state in xi:
        if state in C:
            return -np.inf # log(0)
            
    z_0 = Z_matrix[mdp.start_state, 0]
    if z_0 <= 0: return -np.inf
    
    # R_{w_k}(xi) = sum_{(s,a) in xi} w_k^T phi(s, a)
    rewards = np.dot(mdp.feature_map, w_k)
    path_reward = 0.0
    for i in range(len(xi)-1):
        s, sn = xi[i], xi[i+1]
        a_idx = next(a for a in range(5) if mdp.transitions[s, a] == sn)
        path_reward += rewards[s, a_idx]
    if xi[-1] == mdp.goal_state: 
        path_reward += 10.0
        
    # log(P) = R - log(Z)
    return path_reward - np.log(z_0)

def sample_traj(mdp, weights, Z):
    """Generates an expert demonstration."""
    curr, traj = mdp.start_state, [mdp.start_state]
    rew = np.dot(mdp.feature_map, weights)
    for t in range(mdp.horizon - 1):
        if curr == mdp.goal_state: break
        p = [np.exp(rew[curr, a]) * Z[mdp.transitions[curr, a], t+1] for a in range(5)]
        if sum(p) == 0: break
        curr = mdp.transitions[curr, np.random.choice(5, p=np.array(p)/sum(p))]
        traj.append(curr)
    return traj

# ==========================================
# EM-MLCI ALGORITHM FUNCTIONS
# ==========================================

def identify_candidates(mdp, D):
    """Identify States never visited by any expert in D."""
    visited = set(s for xi in D for s in xi)
    return [s for s in range(mdp.num_states) if s not in visited and s != mdp.goal_state]

def e_step(mdp, D, C_hat, weights, priors):
    """
    E-Step: Calculate the responsibility gamma_{i,k}.
    Math: gamma_{i,k} = (pi_k * P(xi_i | C, w_k)) / sum_{j=1}^K pi_j P(xi_i | C, w_j)
    """
    num_demos = len(D)
    K = len(weights)
    log_gamma = np.zeros((num_demos, K))
    
    Zs = [backward_pass(mdp, weights[k], C_hat) for k in range(K)]
    
    for i, xi_i in enumerate(D):
        for k in range(K):
            log_prob = calculate_trajectory_prob(mdp, xi_i, C_hat, weights[k], Zs[k])
            # log(pi_k * P) = log(pi_k) + log(P)
            log_gamma[i, k] = np.log(priors[k]) + log_prob
            
        # Denominator normalization using LogSumExp for stability
        log_gamma[i, :] -= logsumexp(log_gamma[i, :])
        
    return np.exp(log_gamma) # Convert log probabilities back to standard probabilities

# def m_step_weights(mdp, D, C_hat, weights, responsibilities, lr=0.1, steps=3):
#     """
#     M-Step B: Update Reward Weights (MaxEnt IRL).
#     Math: Nabla_{w_k} L = sum_{i=1}^{|D|} gamma_{i,k} ( phi(xi_i) - E_{P(xi | C, w_k)}[phi(xi)] )
#     """
#     K = len(weights)
#     new_weights = [np.copy(w) for w in weights]
    
#     for k in range(K):
#         for _ in range(steps):
#             Z = backward_pass(mdp, new_weights[k], C_hat)
#             grad = np.zeros(mdp.num_features)
            
#             for i, xi_i in enumerate(D):
#                 if responsibilities[i, k] < 1e-3: continue
                
#                 # Empirical Features phi(xi_i)
#                 emp_counts = np.zeros(mdp.num_features)
#                 for step in range(len(xi_i)-1):
#                     s, sn = xi_i[step], xi_i[step+1]
#                     a = next(a_idx for a_idx in range(5) if mdp.transitions[s, a_idx] == sn)
#                     emp_counts += mdp.feature_map[s, a]
                
#                 # Expected Features E[phi(xi)] (Approximated via sampling)
#                 exp_counts = np.zeros(mdp.num_features)
#                 sample = sample_traj(mdp, new_weights[k], Z)
#                 for step in range(len(sample)-1):
#                     s, sn = sample[step], sample[step+1]
#                     a = next(a_idx for a_idx in range(5) if mdp.transitions[s, a_idx] == sn)
#                     exp_counts += mdp.feature_map[s, a]
                
#                 # Weighted Gradient
#                 grad += responsibilities[i, k] * (emp_counts - exp_counts)
                
#             new_weights[k] += lr * grad / len(D)
            
#     return new_weights

def m_step_weights(mdp, D, C_hat, weights, responsibilities, lr=0.1, steps=5):
    """
    M-Step B: Update Reward Weights (MaxEnt IRL) with stabilized gradients.
    """
    K = len(weights)
    new_weights = [np.copy(w) for w in weights]
    
    for k in range(K):
        for _ in range(steps):
            Z = backward_pass(mdp, new_weights[k], C_hat)
            
            # 1. Compute Expected Features E[phi(xi)] ONCE per cluster step
            exp_counts = np.zeros(mdp.num_features)
            num_samples = 100  # Increase to 100 or more to kill the variance!
            for _ in range(num_samples):
                sample = sample_traj(mdp, new_weights[k], Z)
                for step in range(len(sample)-1):
                    s, sn = sample[step], sample[step+1]
                    a = next(a_idx for a_idx in range(5) if mdp.transitions[s, a_idx] == sn)
                    exp_counts += mdp.feature_map[s, a]
            exp_counts /= num_samples  # Average expected feature counts
            
            # 2. Compute the weighted empirical gradients
            grad = np.zeros(mdp.num_features)
            
            for i, xi_i in enumerate(D):
                if responsibilities[i, k] < 1e-3: continue
                
                # Empirical Features phi(xi_i)
                emp_counts = np.zeros(mdp.num_features)
                for step in range(len(xi_i)-1):
                    s, sn = xi_i[step], xi_i[step+1]
                    a = next(a_idx for a_idx in range(5) if mdp.transitions[s, a_idx] == sn)
                    emp_counts += mdp.feature_map[s, a]
                
                # Gradient: gamma * (Empirical - Expected)
                grad += responsibilities[i, k] * (emp_counts - exp_counts)
                
            # 3. Apply the gradient
            new_weights[k] += lr * grad / len(D)
            
    return new_weights

def calculate_joint_log_likelihood(mdp, D, C, weights, priors):
    """
    Math: L(C, {w_k}, {pi_k}) = sum_{xi in D} log ( sum_{k=1}^K pi_k * P(xi | C, w_k) * I^C(xi) )
    """
    total_log_L = 0
    Zs = [backward_pass(mdp, weights[k], C) for k in range(len(weights))]
    
    for xi in D:
        log_probs = []
        for k in range(len(weights)):
            log_prob = calculate_trajectory_prob(mdp, xi, C, weights[k], Zs[k])
            log_probs.append(np.log(priors[k]) + log_prob)
        # sum_{xi} log( sum_{k} e^{log_probs} )
        total_log_L += logsumexp(log_probs)
    return total_log_L

def m_step_constraints(mdp, D, C_hat, weights, priors, d_DKL):
    """
    M-Step C: Update Constraints.
    Math: Score(c) = sum_{i=1}^{|D|} log( sum_{k=1}^K pi_k * (e^{R_{w_k}(xi_i)} / Z(C U {c}, w_k)) )
    Stops when Delta_{D_{KL}} <= d_DKL
    """
    candidates = identify_candidates(mdp, D)
    current_L = calculate_joint_log_likelihood(mdp, D, C_hat, weights, priors)
    
    while candidates:
        best_c, best_L = None, -np.inf
        
        # Test candidate constraints
        subset = np.random.choice(candidates, min(10, len(candidates)), replace=False)
        for c in subset:
            test_C = C_hat | {c}
            test_L = calculate_joint_log_likelihood(mdp, D, test_C, weights, priors)
            if test_L > best_L:
                best_L = test_L
                best_c = c
        
        # Math: Delta_{D_{KL}} is equivalent to the increase in log-likelihood
        delta_L = best_L - current_L
        
        # Stopping Condition
        if delta_L <= d_DKL:
            break
            
        C_hat.add(best_c)
        candidates.remove(best_c)
        current_L = best_L
        
    return C_hat

def run_em_mlci(mdp, D, K, d_DKL, max_em_iters=10):
    """
    Main loop for Multi-Expert MLCI using Expectation-Maximization.
    """
    # Step 0: Initialization
    C_hat = set()
    num_features = mdp.num_features
    weights = [np.random.randn(num_features) * 0.1 for _ in range(K)]
    priors = np.full(K, 1.0 / K)
    
    for em_iter in range(max_em_iters):
        print(f"--- EM Iteration {em_iter + 1} ---")
        
        # Step 1: E-Step (Expectation)
        responsibilities = e_step(mdp, D, C_hat, weights, priors)
        
        # Step 2: M-Step (Maximization)
        # A. Update Cluster Priors: pi_k = (1/|D|) * sum_{i=1}^{|D|} gamma_{i,k}
        priors = np.mean(responsibilities, axis=0)
        
        # B. Update Reward Weights w_k
        weights = m_step_weights(mdp, D, C_hat, weights, responsibilities)
        
        # C. Update Constraints
        C_hat = m_step_constraints(mdp, D, C_hat, weights, priors, d_DKL)
        
        print(f"Current Inferred Constraints: {sorted(list(C_hat))}")
        
    return C_hat, weights, priors

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
    z1 = backward_pass(mdp, w1, WATER)
    z2 = backward_pass(mdp, w2, WATER)
    
    all_demos = [sample_traj(mdp, w1, z1) for _ in range(N_DEMOS_EXPERT1)] + \
                [sample_traj(mdp, w2, z2) for _ in range(N_DEMOS_EXPERT2)]
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
    # Ensure you have your GridMDP object initialized here
    # Example: mdp = CustomizableFeatureMDP(GRID_SIZE, WATER, GRASS, ROCKS)
    # Example: all_demos = [...]
    w1,w2, WATER, mdp, all_demos, resp = define_mdp_and_demos()
    # Run the EM-MLCI framework
    inferred_c, final_weights, final_priors = run_em_mlci(mdp, all_demos, K=2, d_DKL=0.05, max_em_iters=10)


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

