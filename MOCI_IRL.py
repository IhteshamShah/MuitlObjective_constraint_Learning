import numpy as np
from scipy.special import logsumexp
# ==========================================
# Logic Kernels (MaxEnt IRL)
# ==========================================
# ==========================================
#  MaxEnt Core (Z and Probabilities)
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
# EM-MOCI ALGORITHM FUNCTIONS
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

def calculate_joint_log_likelihood (mdp, D, C, weights, priors):
    """
    Math: L_{avg}(C, {w_k}, {pi_k}) = (1/|D|) * sum_{xi in D} log ( sum_{k=1}^K pi_k * P(xi | C, w_k) * I^C(xi) )
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
        
    # === NEW: Implement L_avg by dividing by dataset size ===
    return total_log_L / len(D)

def calculate_joint_log_likelihood_old (mdp, D, C, weights, priors):
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

def run_em_moci(mdp, D, K, d_DKL, max_em_iters):
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