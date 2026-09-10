import numpy as np
from scipy.special import logsumexp


def _is_action_feasible(mdp, state_idx, action_idx):
    """Return True if a state-action pair is allowed.

    The original MOCI formulation is state-only. This helper adds an optional extension
    for state-action hard constraints while keeping the legacy implementation unchanged
    when no feasibility mask is defined.
    """
    if not hasattr(mdp, "action_feasible_mask") or mdp.action_feasible_mask is None:
        return True

    mask = mdp.action_feasible_mask
    if callable(mask):
        return bool(mask(state_idx, action_idx, mdp))
    if isinstance(mask, np.ndarray):
        return bool(mask[state_idx, action_idx])
    if isinstance(mask, list):
        return bool(mask[state_idx][action_idx])
    if isinstance(mask, dict):
        return bool(mask.get((state_idx, action_idx), True))
    return True


def _is_forbidden_constraint(state_idx, action_idx, constraint_set):
    """Check if a state or state-action pair is forbidden under the constraint set."""
    if state_idx in constraint_set:
        return True
    if action_idx is not None and (state_idx, action_idx) in constraint_set:
        return True
    return False

# ==========================================
# Logic Kernels (MaxEnt IRL)
# ==========================================
# ==========================================
#  MaxEnt Core (Z and Probabilities)
# ==========================================
def backward_pass(mdp, weights, obstacles):
    """Compute log partition matrix logZ(C, w) with numerically stable recursion."""
    logZ = np.full((mdp.num_states, mdp.horizon + 1), -np.inf, dtype=float)
    rewards = np.dot(mdp.feature_map, weights)

    state_constraints = {c for c in obstacles if isinstance(c, (int, np.integer))}
    action_constraints = {c for c in obstacles if isinstance(c, tuple)}

    # At horizon, non-goal terminal contribution is log(1)=0.
    logZ[:, mdp.horizon] = 0.0
    logZ[mdp.goal_state, mdp.horizon] = 10.0
    for obs in state_constraints:
        logZ[obs, :] = -np.inf

    for t in range(mdp.horizon - 1, -1, -1):
        for s in range(mdp.num_states):
            if s in state_constraints:
                continue
            if s == mdp.goal_state:
                logZ[s, t] = 10.0 + logZ[s, t + 1]
                continue

            terms = []
            for a in range(mdp.num_actions):
                if (s, a) in action_constraints:
                    continue
                if not _is_action_feasible(mdp, s, a):
                    continue
                sn = mdp.transitions[s, a]
                next_val = logZ[sn, t + 1]
                if np.isfinite(next_val):
                    terms.append(rewards[s, a] + next_val)

            if terms:
                logZ[s, t] = logsumexp(np.array(terms, dtype=float))

    return logZ

def calculate_trajectory_prob(mdp, xi, C, w_k, logZ_matrix):
    """
    Math: P(xi | C, w_k) = (e^{R_{w_k}(xi)} / Z(C, w_k)) * I^C(xi)
    Returns the log probability for stability. Returns -inf if impossible.
    """
    # Indicator function I^C(xi): accept both state-only and state-action constraints
    for state in xi:
        if state in C:
            return -np.inf

    rewards = np.dot(mdp.feature_map, w_k)
    path_reward = 0.0
    # Fixed penalty for transitions not representable by the compressed action model.
    # This avoids total likelihood collapse on large observational MDPs.
    unmatched_log_prob = np.log(1e-12)
    for i in range(len(xi) - 1):
        s, sn = xi[i], xi[i+1]
        a_idx = None
        for a in range(mdp.num_actions):
            if _is_forbidden_constraint(s, a, C):
                continue
            if _is_action_feasible(mdp, s, a) and mdp.transitions[s, a] == sn:
                a_idx = a
                break
        if a_idx is None:
            path_reward += unmatched_log_prob
        else:
            path_reward += rewards[s, a_idx]

    log_z_0 = logZ_matrix[mdp.start_state, 0]
    if not np.isfinite(log_z_0):
        return -np.inf
    if xi[-1] == mdp.goal_state:
        path_reward += 10.0

    # log(P) = R - log(Z)
    return path_reward - log_z_0

def sample_traj(mdp, weights, logZ, constraints=None):
    """Generates an expert demonstration."""
    curr, traj = mdp.start_state, [mdp.start_state]
    rew = np.dot(mdp.feature_map, weights)
    if constraints is None:
        constraints = set()

    for t in range(mdp.horizon - 1):
        if curr == mdp.goal_state: break
        feasible_actions = []
        for a in range(mdp.num_actions):
            if _is_action_feasible(mdp, curr, a) and not _is_forbidden_constraint(curr, a, constraints):
                feasible_actions.append(a)
        if not feasible_actions:
            break
        logp = []
        valid_actions = []
        for a in feasible_actions:
            sn = mdp.transitions[curr, a]
            next_log = logZ[sn, t + 1]
            if np.isfinite(next_log):
                valid_actions.append(a)
                logp.append(rew[curr, a] + next_log)
        if not logp:
            break

        logp = np.array(logp, dtype=float)
        probs = np.exp(logp - logsumexp(logp))
        action_idx = np.random.choice(valid_actions, p=probs)
        curr = mdp.transitions[curr, action_idx]
        traj.append(curr)
    return traj

# ==========================================
# EM-MOCI ALGORITHM FUNCTIONS
# ==========================================

def identify_candidates(mdp, D):
    """Identify states or state-action pairs never used by the experts in D."""
    if getattr(mdp, "constraint_mode", "state") == "state_action":
        candidate_pairs = []
        used_pairs = set()
        for xi in D:
            for i in range(len(xi) - 1):
                s, sn = xi[i], xi[i+1]
                feasible_action_candidates = [
                    a for a in range(mdp.num_actions)
                    if _is_action_feasible(mdp, s, a) and mdp.transitions[s, a] == sn
                ]
                used_pairs.update((s, a) for a in feasible_action_candidates)
        for s in range(mdp.num_states):
            if s == mdp.goal_state:
                continue
            for a in range(mdp.num_actions):
                if _is_action_feasible(mdp, s, a) and (s, a) not in used_pairs:
                    candidate_pairs.append((s, a))
        return candidate_pairs

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
    safe_priors = np.clip(priors, 1e-12, None)
    safe_priors /= safe_priors.sum()

    Zs = [backward_pass(mdp, weights[k], C_hat) for k in range(K)]

    for i, xi_i in enumerate(D):
        for k in range(K):
            log_prob = calculate_trajectory_prob(mdp, xi_i, C_hat, weights[k], Zs[k])
            # log(pi_k * P) = log(pi_k) + log(P)
            log_gamma[i, k] = np.log(safe_priors[k]) + log_prob

        # Denominator normalization using LogSumExp for stability
        finite = np.isfinite(log_gamma[i, :])
        if not np.any(finite):
            log_gamma[i, :] = -np.inf
            continue
        log_gamma[i, finite] -= logsumexp(log_gamma[i, finite])

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
                sample = sample_traj(mdp, new_weights[k], Z, C_hat)
                for step in range(len(sample)-1):
                    s, sn = sample[step], sample[step+1]
                    feasible_actions = [
                        a_idx for a_idx in range(mdp.num_actions)
                        if _is_action_feasible(mdp, s, a_idx)
                        and not _is_forbidden_constraint(s, a_idx, C_hat)
                        and mdp.transitions[s, a_idx] == sn
                    ]
                    if not feasible_actions:
                        continue
                    a = feasible_actions[0]
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
                    feasible_actions = [
                        a_idx for a_idx in range(mdp.num_actions)
                        if _is_action_feasible(mdp, s, a_idx)
                        and not _is_forbidden_constraint(s, a_idx, C_hat)
                        and mdp.transitions[s, a_idx] == sn
                    ]
                    if not feasible_actions:
                        return new_weights
                    a = feasible_actions[0]
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
    safe_priors = np.clip(priors, 1e-12, None)
    safe_priors /= safe_priors.sum()

    for xi in D:
        log_probs = []
        for k in range(len(weights)):
            log_prob = calculate_trajectory_prob(mdp, xi, C, weights[k], Zs[k])
            log_probs.append(np.log(safe_priors[k]) + log_prob)
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
    safe_priors = np.clip(priors, 1e-12, None)
    safe_priors /= safe_priors.sum()

    for xi in D:
        log_probs = []
        for k in range(len(weights)):
            log_prob = calculate_trajectory_prob(mdp, xi, C, weights[k], Zs[k])
            log_probs.append(np.log(safe_priors[k]) + log_prob)
        # sum_{xi} log( sum_{k} e^{log_probs} )
        total_log_L += logsumexp(log_probs)
    return total_log_L

def m_step_constraints(mdp, D, C_hat, weights, priors, d_DKL, candidate_subset_size=None):
    """
    M-Step C: Update Constraints.
    Math: Score(c) = sum_{i=1}^{|D|} log( sum_{k=1}^K pi_k * (e^{R_{w_k}(xi_i)} / Z(C U {c}, w_k)) )
    Stops when Delta_{D_{KL}} <= d_DKL
    """
    candidates = identify_candidates(mdp, D)
    current_L = calculate_joint_log_likelihood(mdp, D, C_hat, weights, priors)
    
    while candidates:
        best_c, best_L = None, -np.inf

        # Test candidate constraints. By default evaluate all candidates to reduce
        # stochastic variability in learned constraints.
        if candidate_subset_size is None or candidate_subset_size >= len(candidates):
            subset = list(candidates)
        elif candidate_subset_size <= 0:
            subset = list(candidates)
        else:
            subset = list(candidates)[: int(candidate_subset_size)]

        for c in subset:
            test_C = C_hat | {c}
            test_L = calculate_joint_log_likelihood(mdp, D, test_C, weights, priors)
            if test_L > best_L:
                best_L = test_L
                best_c = c
        
        # Math: Delta_{D_{KL}} is equivalent to the increase in log-likelihood
        if best_c is None:
            break

        delta_L = best_L - current_L

        # Stopping Condition
        if delta_L <= d_DKL:
            break

        C_hat.add(best_c)
        if best_c in candidates:
            candidates.remove(best_c)
        current_L = best_L
        
    return C_hat

def run_em_moci(mdp, D, K, d_DKL, max_em_iters, candidate_subset_size=None):
    """
    Main loop for Multi-Expert MLCI using Expectation-Maximization.
    """
    # Step 0: Initialization
    C_hat = set()
    num_features = mdp.num_features
    weights = [np.random.randn(num_features) * 0.1 for _ in range(K)]
    priors = np.full(K, 1.0 / K)
    if hasattr(mdp, "constraint_mode") and mdp.constraint_mode == "state_action":
        C_hat = set()

    for em_iter in range(max_em_iters):
        print(f"--- EM Iteration {em_iter + 1} ---")
        
        # Step 1: E-Step (Expectation)
        responsibilities = e_step(mdp, D, C_hat, weights, priors)
        
        # Step 2: M-Step (Maximization)
        # A. Update Cluster Priors: pi_k = (1/|D|) * sum_{i=1}^{|D|} gamma_{i,k}
        priors = np.mean(responsibilities, axis=0)
        priors = np.clip(priors, 1e-12, None)
        priors /= priors.sum()

        # B. Update Reward Weights w_k
        weights = m_step_weights(mdp, D, C_hat, weights, responsibilities)
        
        # C. Update Constraints
        C_hat = m_step_constraints(mdp, D, C_hat, weights, priors, d_DKL, candidate_subset_size=candidate_subset_size)
        
        print(f"Current Inferred Constraints: {sorted(list(C_hat))}")
        
    return C_hat, weights, priors