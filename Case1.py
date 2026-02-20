import numpy as np
from scipy.special import logsumexp

# ==========================================
# 1. Custom Feature Gridworld MDP
# ==========================================
class CustomizableFeatureMDP:
    def __init__(self, size, water_states, grass_states, rock_states, horizon=20):
        self.size = size
        self.num_states = size * size
        self.num_actions = 5  # U, D, L, R, Stay
        self.horizon = horizon
        self.start_state = 0
        self.goal_state = self.num_states - 1
        
        self.num_features = 4 # 0:Sand, 1:Grass, 2:Rock, 3:Water
        self.feature_grid = np.zeros(self.num_states, dtype=int)
        
        for s in grass_states: self.feature_grid[s] = 1
        for s in rock_states:  self.feature_grid[s] = 2
        for s in water_states: self.feature_grid[s] = 3
        
        self.transitions = np.zeros((self.num_states, self.num_actions), dtype=int)
        for r in range(size):
            for c in range(size):
                s = r * size + c
                self.transitions[s, 0] = max(0, r-1) * size + c # Up
                self.transitions[s, 1] = min(size-1, r+1) * size + c # Down
                self.transitions[s, 2] = r * size + max(0, c-1) # Left
                self.transitions[s, 3] = r * size + min(size-1, c+1) # Right
                self.transitions[s, 4] = s # Stay

        # Feature Map (S, A, F)
        self.feature_map = np.zeros((self.num_states, self.num_actions, self.num_features))
        for s in range(self.num_states):
            self.feature_map[s, :, self.feature_grid[s]] = 1.0

# ==========================================
# 2. MaxEnt IRL Core Operations
# ==========================================
def backward_pass(mdp, weights, obstacles):
    """Computes Partition Function Z in log-space to prevent underflow."""
    Z = np.zeros((mdp.num_states, mdp.horizon + 1))
    rewards = np.dot(mdp.feature_map, weights)
    
    Z[:, mdp.horizon] = 1.0
    Z[mdp.goal_state, mdp.horizon] = np.exp(10.0) # Goal bonus
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

def get_log_likelihood(demo, mdp, weights, Z):
    """Calculates log P(xi | C, w_k) safely."""
    if Z[mdp.start_state, 0] <= 0: return -1e9
    rewards = np.dot(mdp.feature_map, weights)
    path_reward = 0.0
    for i in range(len(demo)-1):
        s, sn = demo[i], demo[i+1]
        a_idx = next(a for a in range(5) if mdp.transitions[s, a] == sn)
        path_reward += rewards[s, a_idx]
    if demo[-1] == mdp.goal_state: path_reward += 10.0
    return path_reward - np.log(Z[mdp.start_state, 0])

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
# 3. EM-MLCI Pipeline
# ==========================================
def identify_candidates(mdp, D):
    """States not visited in any demonstration are candidate constraints."""
    visited = set(s for d in D for s in d)
    return [s for s in range(mdp.num_states) if s not in visited and s != mdp.goal_state]

def e_step(mdp, D, C_hat, weights, priors):
    """Computes responsibilities gamma_ik using log-probabilities."""
    num_demos = len(D)
    K = len(weights)
    log_gamma = np.zeros((num_demos, K))
    
    # Precompute Z for each cluster
    Zs = [backward_pass(mdp, weights[k], C_hat) for k in range(K)]
    
    for i, xi_i in enumerate(D):
        for k in range(K):
            ll = get_log_likelihood(xi_i, mdp, weights[k], Zs[k])
            log_gamma[i, k] = np.log(priors[k]) + ll
            
        # Log-Sum-Exp trick for numerical stability
        log_gamma[i, :] -= logsumexp(log_gamma[i, :])
        
    return np.exp(log_gamma) # Convert back to normal probabilities [0, 1]

def update_weights(mdp, D, C_hat, weights, gamma, lr=0.1, steps=3):
    """Simplified MaxEnt IRL M-Step: Gradient ascent on weighted log-likelihood."""
    K = len(weights)
    new_weights = [np.copy(w) for w in weights]
    
    for k in range(K):
        for _ in range(steps):
            Z = backward_pass(mdp, new_weights[k], C_hat)
            grad = np.zeros(mdp.num_features)
            
            # For each demo, weight the empirical features by responsibility gamma[i,k]
            for i, demo in enumerate(D):
                if gamma[i, k] < 1e-3: continue # Skip if responsibility is practically zero
                
                # Empirical counts
                emp_counts = np.zeros(mdp.num_features)
                for step in range(len(demo)-1):
                    s, sn = demo[step], demo[step+1]
                    a = next(a_idx for a_idx in range(5) if mdp.transitions[s, a_idx] == sn)
                    emp_counts += mdp.feature_map[s, a]
                
                # Approximate Expected counts by sampling from current weights
                exp_counts = np.zeros(mdp.num_features)
                sample = sample_traj(mdp, new_weights[k], Z)
                for step in range(len(sample)-1):
                    s, sn = sample[step], sample[step+1]
                    a = next(a_idx for a_idx in range(5) if mdp.transitions[s, a_idx] == sn)
                    exp_counts += mdp.feature_map[s, a]
                
                # Gradient update weighted by responsibility
                grad += gamma[i, k] * (emp_counts - exp_counts)
                
            new_weights[k] += lr * grad / len(D)
            
    return new_weights

def calculate_joint_log_likelihood(mdp, D, C, weights, priors):
    """Calculates Total L = sum_i log( sum_k pi_k * P(xi_i | C, w_k) ) safely."""
    total_log_L = 0
    Zs = [backward_pass(mdp, weights[k], C) for k in range(len(weights))]
    
    for xi in D:
        log_probs = []
        for k in range(len(weights)):
            ll = get_log_likelihood(xi, mdp, weights[k], Zs[k])
            log_probs.append(np.log(priors[k]) + ll)
        total_log_L += logsumexp(log_probs)
    return total_log_L

def run_greedy_mlci(mdp, D, C_hat, weights, priors, d_DKL):
    """Iteratively infers shared constraints using KL-divergence stopping criteria."""
    candidates = identify_candidates(mdp, D)
    current_L = calculate_joint_log_likelihood(mdp, D, C_hat, weights, priors)
    
    while candidates:
        best_c, best_L = None, -np.inf
        
        # Test a subset of candidates for performance (e.g., 10)
        subset = np.random.choice(candidates, min(10, len(candidates)), replace=False)
        for c in subset:
            test_C = C_hat | {c}
            test_L = calculate_joint_log_likelihood(mdp, D, test_C, weights, priors)
            if test_L > best_L:
                best_L = test_L
                best_c = c
        
        delta_L = best_L - current_L
        
        # STOPPING CRITERION: Theorem 1 / Algorithm 2
        # Delta_L is the improvement in joint log-likelihood, which equates to D_KL reduction
        if delta_L <= d_DKL:
            break
            
        C_hat.add(best_c)
        candidates.remove(best_c)
        current_L = best_L
        
    return C_hat

# ==========================================
# 4. Main Execution
# ==========================================
def main(mdp, D):
    C_hat = set()
    d_DKL = 0.05   # Stopping Threshold
    K = 2         # Number of expert clusters
    max_em_iters = 100
    
    # Initialize random weights for the 4 features
    weights = [np.random.randn(mdp.num_features) * 0.1 for _ in range(K)]
    priors = np.full(K, 1.0 / K)
    
    print("Starting Latent Preference EM-MLCI...")
    for iteration in range(max_em_iters):
        print(f"\n--- EM Iteration {iteration + 1} ---")
        
        # E-Step
        gamma = e_step(mdp, D, C_hat, weights, priors)
        
        # M-Step A: Priors
        priors = np.mean(gamma, axis=0)
        
        # M-Step B: Weights (MaxEnt IRL)
        weights = update_weights(mdp, D, C_hat, weights, gamma)
        
        # M-Step C: Constraints
        old_C_len = len(C_hat)
        C_hat = run_greedy_mlci(mdp, D, C_hat, weights, priors, d_DKL)
        
        print(f"Current Priors: {priors}")
        print(f"Inferred Constraints: {sorted(list(C_hat))}")
        
        # Early stopping for EM if constraints stop changing
        if len(C_hat) == old_C_len and iteration > 1:
            print("Convergence reached.")
            break

    print(f"\n=== Final Inferred Constraints: {sorted(list(C_hat))} ===")
    return C_hat

if __name__ == "__main__":
    # --- SETUP ENVIRONMENT ---
    GRID_SIZE = 7 
    WATER = [15, 16, 17, 24] # Ground Truth Shared Constraints
    GRASS = [2, 3, 9, 10, 22, 23, 29, 30]
    ROCKS = [4, 5, 11, 12, 25, 26, 32, 33, 39, 40]
    
    mdp = CustomizableFeatureMDP(GRID_SIZE, WATER, GRASS, ROCKS)
    
    # --- GENERATE SYNTHETIC EXPERT DEMOS ---
    w1 = np.array([0.0, 5.0, 0.0, -10.0]) # Expert 1 loves Grass
    w2 = np.array([0.0, 0.0, 5.0, -10.0]) # Expert 2 loves Rocks
    
    z1 = backward_pass(mdp, w1, WATER)
    z2 = backward_pass(mdp, w2, WATER)
    
    all_demos = [sample_traj(mdp, w1, z1) for _ in range(10)] + \
                [sample_traj(mdp, w2, z2) for _ in range(10)]
    
    # --- RUN INFERENCE ---
    main(mdp, all_demos)
























# import numpy as np
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import gridworld as gw
# from MOCL_IRL import backward_pass, sample_traj, get_log_likelihood


# def calculate_trajectory_prob(mdp, demo, C, weight_k):
#     """
#     Calculates P(xi | C, w_k) for a single demonstration.
    
#     Inputs:
#         mdp: The MDP environment object.
#         demo: A list of states representing the trajectory xi.
#         C: The current set of inferred constraints (set or list of states).
#         weight_k: The reward weights for expert cluster k.
        
#     Outcome:
#         A float representing the probability of the trajectory. 
#         Returns 0.0 if the trajectory violates constraints.
#     """
#     # 1. Indicator Function Check (I^C(xi))
#     # If the demonstration hits any constraint in C, it is impossible.
#     for state in demo:
#         if state in C:
#             return 0.0
            
#     # 2. Compute Partition Function Z(C, w_k)
#     # This runs the backward_pass to get Z values for all states
#     Z_matrix = backward_pass(mdp, weight_k, C)
    
#     # The total partition function for the environment is Z at the start state at t=0
#     z_0 = Z_matrix[mdp.start_state, 0]
    
#     # If z_0 is 0, there are no valid paths to the goal under these constraints
#     if z_0 <= 0:
#         return 0.0
        
#     # 3. Calculate Trajectory Reward R_{w_k}(xi)
#     # Multiply the environment's feature map by the expert's weights
#     rewards = np.dot(mdp.feature_map, weight_k)
#     path_reward = 0.0
    
#     for i in range(len(demo) - 1):
#         s = demo[i]
#         s_next = demo[i+1]
        
#         # Find which action was taken to move from s to s_next
#         action_taken = 4 # Default to action 4 (Stay)
#         for a in range(mdp.num_actions):
#             if mdp.transitions[s, a] == s_next:
#                 action_taken = a
#                 break
                
#         path_reward += rewards[s, action_taken]
        
#     # Add the terminal reward if the agent reached the goal state
#     if demo[-1] == mdp.goal_state:
#         path_reward += rewards[mdp.goal_state, 4]
        
#     # 4. Compute Final Probability
#     # P = exp(R) / Z
#     # Note: To avoid numerical overflow with exp(), it's common practice to 
#     # calculate this as exp(path_reward - log(z_0))
#     prob = np.exp(path_reward - np.log(z_0))
    
#     return prob

# def run_em_mlci(mdp, D, K, d_DKL, max_em_iters=10):
#     """
#     Main loop for Multi-Expert MLCI using Expectation-Maximization.
    
#     Inputs:
#         mdp: The nominal MDP model.
#         D: Set of demonstrated trajectories[cite: 507].
#         K: Number of latent expert clusters.
#         d_DKL: Stopping threshold for KL-divergence improvement[cite: 601].
#         max_em_iters: Maximum iterations for the EM loop.
        
#     Outcome:
#         C_hat: Final inferred shared constraint set.
#         weights: Learned reward weights w_k for each cluster.
#         priors: Learned cluster priors pi_k.
#     """
#     # Step 0: Initialization [cite: 507]
#     C_hat = set()
#     num_features = mdp.num_features
#     weights = [np.random.randn(num_features) for _ in range(K)]
#     priors = np.full(K, 1.0 / K)
    
#     for em_iter in range(max_em_iters):
#         # Step 1: E-Step (Expectation)
#         # Calculate gamma_ik = pi_k * P(xi_i | C, w_k) / sum(pi_j * P(xi_i | C, w_j))
#         responsibilities = e_step(mdp, D, C_hat, weights, priors)
        
#         # Step 2: M-Step (Maximization)
#         # A. Update Cluster Priors pi_k = (1/|D|) * sum(gamma_ik)
#         priors = np.mean(responsibilities, axis=0)
        
#         # B. Update Reward Weights w_k using MaxEnt IRL Gradient Descent
#         # Gradient: sum_i gamma_ik * (phi(xi_i) - E[phi])
#         weights = m_step_weights(mdp, D, C_hat, weights, responsibilities)
        
#         # C. Update Constraints (Shared MLCI Core) [cite: 593, 594]
#         # Iteratively add constraints until KL-divergence improvement < d_DKL
#         C_hat = m_step_constraints(mdp, D, C_hat, weights, priors, d_DKL)
        
#     return C_hat, weights, priors

# def e_step(mdp, D, C_hat, weights, priors):
#     """
#     Calculates posterior probabilities (responsibilities) for each demo.
#     Outcome: array of shape (num_demos, K)
#     """
#     num_demos = len(D)
#     K = len(weights)
#     gamma = np.zeros((num_demos, K))
    
#     for i, xi_i in enumerate(D):
#         for k in range(K):
#             # P(xi | k, C, w_k) = (e^R(xi) / Z(C, w_k)) * I(xi) [cite: 546, 550]
#             prob = calculate_trajectory_prob(mdp, xi_i, C_hat, weights[k])
#             gamma[i, k] = priors[k] * prob
            
#         # Normalize: gamma_ik = numerator / sum_j(numerator_j)
#         gamma[i, :] /= np.sum(gamma[i, :])
#     return gamma

# def m_step_constraints(mdp, D, C_hat, weights, priors, d_DKL):
#     """
#     Greedy Iterative Constraint Inference adapted for multiple experts.
#     Outcome: Updated shared constraint set C_hat.
#     """
#     K = len(weights)
#     candidates = identify_candidates(mdp, D) # States never visited
    
#     while True:
#         best_c = None
#         best_likelihood = -np.inf
        
#         # Current joint log-likelihood: sum_i log(sum_k pi_k * P(xi_i | C, w_k))
#         current_L = calculate_joint_log_likelihood(mdp, D, C_hat, weights, priors)
        
#         for c in candidates:
#             if c in C_hat: continue
            
#             # Score(c) = sum_i log(sum_k pi_k * (e^R(xi_i) / Z(C U {c}, w_k))) [cite: 550, 855]
#             temp_C = C_hat | {c}
#             test_L = calculate_joint_log_likelihood(mdp, D, temp_C, weights, priors)
            
#             if test_L > best_likelihood:
#                 best_likelihood = test_L
#                 best_c = c
        
#         # Math: Delta_DKL = DKL(P_D || P_M_C) - DKL(P_D || P_M_C_U_c) [cite: 601]
#         # This is equivalent to checking improvement in log-likelihood
#         delta_L = best_likelihood - current_L
        
#         # Stopping Condition: if Delta_DKL <= d_DKL then break [cite: 601]
#         if delta_L <= d_DKL:
#             break
            
#         C_hat.add(best_c)
#     return C_hat

# def calculate_joint_log_likelihood(mdp, D, C, weights, priors):
#     """
#     Computes L = sum_i log( sum_k pi_k * P(xi_i | C, w_k) )
#     Outcome: Scalar log-likelihood value.
#     """
#     total_log_L = 0
#     for xi in D:
#         weighted_sum_probs = 0
#         for k in range(len(weights)):
#             # P(xi | C, w_k) = e^R(xi) / Z(C, w_k) [cite: 546]
#             prob = calculate_trajectory_prob(mdp, xi, C, weights[k])
#             weighted_sum_probs += priors[k] * prob
#         total_log_L += np.log(weighted_sum_probs)
#     return total_log_L



# import numpy as np

# def main(mdp, D):
#     # --- Step 0: Initialization ---
#     # Constraints (C_hat): Start with an empty set 
#     # d_DKL: Threshold to avoid overfitting (e.g., 0.1) [cite: 149, 169]
#     C_hat = set()
#     d_DKL = 0.1
#     K = 2  # Number of expert clusters
#     max_em_iters = 100 #  Maximum EM iterations to ensure convergence
    
#     # Initialize weights w_k and priors pi_k
#     weights = [np.random.randn(mdp.num_features) for _ in range(K)]
#     priors = np.full(K, 1.0 / K)
    
#     # EM Loop
#     for iteration in range(max_em_iters):
#         # Step 1: E-Step (Expectation)
#         # Calculate responsibilities gamma_ik for each demonstration
#         # Formula: gamma_ik = (pi_k * P(xi_i | C, w_k)) / sum(pi_j * P(xi_i | C, w_j))
#         gamma = e_step(mdp, D, C_hat, weights, priors)
        
#         # Step 2: M-Step (Maximization)
#         # A. Update Cluster Priors
#         # Formula: pi_k = (1 / |D|) * sum_i(gamma_ik)
#         priors = np.mean(gamma, axis=0)
        
#         # B. Update Reward Weights (MaxEnt IRL)
#         # Gradient: sum_i gamma_ik * (phi(xi_i) - E[phi | C, w_k])
#         weights = update_weights(mdp, D, C_hat, weights, gamma)
        
#         # C. Update Constraints (Shared MLCI Core - Algorithm 2)
#         # Iteratively add constraints that maximize joint log-likelihood
#         # Score(c) = sum_i log(sum_k pi_k * P(xi_i | C U {c}, w_k))
#         C_hat = run_greedy_mlci(mdp, D, C_hat, weights, priors, d_DKL)

#     print(f"Final Inferred Constraints: {C_hat}")

# if __name__ == "__main__":
#     # --- STEP 1: DEFINE GRIDWORLD SIZE ---
#     GRID_SIZE = 7 

#     # --- STEP 2: DEFINE TERRAIN STATES (indices) ---
#     WATER = [15, 16, 17, 18, 24, 31] # RIVER / HARD CONSTRAINTS
#     GRASS = [2, 3, 9, 10, 22, 23, 29, 30]
#     ROCKS = [4, 5, 11, 12, 25, 26, 32, 33, 39, 40]

#     # --- STEP 3: DEFINE DEMONSTRATION COUNTS ---
#     N_DEMOS_EXPERT1 = 10
#     N_DEMOS_EXPERT2 = 10
#     mdp = gw.CustomizableFeatureMDP(GRID_SIZE, WATER, GRASS, ROCKS)
    
#     # Define Preferences [Sand, Grass, Rock, Water]
#     w1 = np.array([0.0, 3.0, 0.0, -10.0]) # Expert 1: Grass Lover
#     w2 = np.array([0.0, 0.0, 3.0, -10.0]) # Expert 2: Rock Lover
    
#     # Generate Demos
#     z1 = backward_pass(mdp, w1, WATER)
#     z2 = backward_pass(mdp, w2, WATER)
    
#     all_demos = [sample_traj(mdp, w1, z1) for _ in range(N_DEMOS_EXPERT1)] + \
#                 [sample_traj(mdp, w2, z2) for _ in range(N_DEMOS_EXPERT2)]
#     # Mock responsibilities for visualization
#     main(mdp, all_demos)