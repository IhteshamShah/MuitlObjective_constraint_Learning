import numpy as np
import copy

# ==========================================
# 1. CORE MATH FOR MAX-ENT IRL (BACKWARD / FORWARD PASS)
# ==========================================

def compute_soft_policy(mdp):
    """
    Backward Pass (Soft Value Iteration)
    Computes the Maximum Entropy policy given the MDP's current rewards and constraints.
    """
    Q = np.zeros((mdp.num_states, mdp.num_actions))
    V = np.zeros(mdp.num_states)
    
    for _ in range(mdp.horizon):
        Q_new = np.zeros((mdp.num_states, mdp.num_actions))
        for s in range(mdp.num_states):
            # If the state is a constraint, its value is strongly negative
            if s in mdp.constraints:
                V[s] = -1e9
                continue
                
            for a in range(mdp.num_actions):
                # Simple deterministic transition for this example
                s_prime = mdp.get_transition(s, a)
                reward = mdp.rewards[s]
                Q_new[s, a] = reward + mdp.gamma * V[s_prime]
                
        # Softmax for V(s) = log(sum(exp(Q(s,a))))
        # Subtract max for numerical stability
        for s in range(mdp.num_states):
            if s not in mdp.constraints:
                max_q = np.max(Q_new[s])
                V[s] = max_q + np.log(np.sum(np.exp(Q_new[s] - max_q)))
        Q = Q_new

    # Compute Policy: pi(a|s) = exp(Q(s,a) - V(s))
    policy = np.zeros((mdp.num_states, mdp.num_actions))
    for s in range(mdp.num_states):
        if s not in mdp.constraints:
            policy[s] = np.exp(Q[s] - V[s])
            # Normalize to handle minor floating point errors
            policy[s] /= np.sum(policy[s]) 
            
    return policy

def compute_state_visitation_freq(mdp, policy):
    """
    Forward Pass.
    Computes the expected state visitations (q) using the policy.
    """
    D = np.zeros((mdp.horizon, mdp.num_states))
    
    # Assume uniform start state distribution for non-constraint states
    valid_starts = [s for s in range(mdp.num_states) if s not in mdp.constraints]
    for s in valid_starts:
        D[0, s] = 1.0 / len(valid_starts)
        
    for t in range(1, mdp.horizon):
        for s in range(mdp.num_states):
            if s in mdp.constraints:
                continue
            for a in range(mdp.num_actions):
                s_prime = mdp.get_transition(s, a)
                if s_prime not in mdp.constraints:
                    D[t, s_prime] += D[t-1, s] * policy[s, a]
                    
    # Sum over all timesteps
    expected_visitations = np.sum(D, axis=0)
    # Normalize to create a valid probability distribution
    return expected_visitations / (np.sum(expected_visitations) + 1e-9)

# ==========================================
# 2. THE MLCI ALGORITHM
# ==========================================

def calculate_kl_divergence(expert_trajs, mdp):
    """Calculates D_KL between empirical expert data and MDP expectations."""
    # 1. Calculate Empirical Distribution (p)
    total_states = mdp.num_states
    empirical_counts = np.zeros(total_states)
    total_visits = 0
    for traj in expert_trajs:
        for state, _ in traj:
            empirical_counts[state] += 1
            total_visits += 1
            
    p = empirical_counts / max(total_visits, 1) # Prevent division by zero

    # 2. Calculate Expected Distribution (q)
    q = compute_state_visitation_freq(mdp, mdp.policy) 
    
    # 3. Compute KL Divergence: sum( p * log(p / q) )
    # We only compute where p > 0 to avoid -inf, and add epsilon to q to avoid log(inf)
    kl = 0.0
    for s in range(total_states):
        if p[s] > 0:
            kl += p[s] * np.log(p[s] / (q[s] + 1e-9))
    return kl

def run_mlci_inference(nominal_mdp, expert_trajs, d_kl_threshold=0.01, max_constraints=20):
    """Greedy Maximum Likelihood Constraint Inference"""
    estimated_constraints = set()
    total_states = nominal_mdp.num_states
    
    # Ensure policy is updated initially
    nominal_mdp.update_policy()
    
    current_mdp = copy.deepcopy(nominal_mdp)
    kl_history = [calculate_kl_divergence(expert_trajs, current_mdp)]
    print(f"Initial KL Divergence: {kl_history[0]:.4f}")

    # Identify unvisited states
    visited_states = {state for traj in expert_trajs for state, _ in traj}
    candidate_constraints = [s for s in range(total_states) if s not in visited_states]

    for i in range(max_constraints):
        best_candidate = None
        best_kl_drop = -np.inf
        temp_best_mdp = None

        print(f"Testing {len(candidate_constraints) - len(estimated_constraints)} candidates...")
        for candidate in candidate_constraints:
            if candidate in estimated_constraints:
                continue
            
            test_mdp = copy.deepcopy(current_mdp)
            test_constraints = estimated_constraints | {candidate}
            test_mdp.set_constraints(test_constraints)
            
            new_kl = calculate_kl_divergence(expert_trajs, test_mdp)
            kl_drop = kl_history[-1] - new_kl
            
            if kl_drop > best_kl_drop:
                best_kl_drop = kl_drop
                best_candidate = candidate
                temp_best_mdp = test_mdp

        if best_candidate is None or best_kl_drop <= 0:
            print("No more constraints improve the likelihood.")
            break

        estimated_constraints.add(best_candidate)
        kl_history.append(calculate_kl_divergence(expert_trajs, temp_best_mdp))
        current_mdp = temp_best_mdp

        print(f"Iteration {i+1}: Added state {best_candidate} as constraint. KL dropped to: {kl_history[-1]:.4f}")

        # Stopping condition
        if abs(kl_history[-2] - kl_history[-1]) < d_kl_threshold:
            print(f"Stopping due to KL convergence (Change < {d_kl_threshold}).")
            estimated_constraints.remove(best_candidate)
            break

    return estimated_constraints

# ==========================================
# 3. MOCK ENVIRONMENT & EXECUTION
# ==========================================

class SimpleGridMDP:
    """A lightweight MDP for testing the algorithm."""
    def __init__(self, size=5):
        self.size = size
        self.num_states = size * size
        self.num_actions = 4 # 0:Up, 1:Down, 2:Left, 3:Right
        self.horizon = 15
        self.gamma = 0.95
        
        self.rewards = np.zeros(self.num_states)
        self.rewards[-1] = 10.0 # Goal state at bottom right
        
        self.constraints = set()
        self.policy = None

    def get_transition(self, s, a):
        x, y = s % self.size, s // self.size
        if a == 0 and y > 0: y -= 1
        elif a == 1 and y < self.size - 1: y += 1
        elif a == 2 and x > 0: x -= 1
        elif a == 3 and x < self.size - 1: x += 1
        return y * self.size + x
        
    def set_constraints(self, constraints):
        self.constraints = constraints
        self.update_policy()
        
    def update_policy(self):
        self.policy = compute_soft_policy(self)

def run_algo(mock_trajs, true_constraints, size):
    # 1. Initialize Nominal MDP (5x5 grid)
    mdp = SimpleGridMDP(size)
    
    # # 2. Define "True" Hard Constraints (e.g., states 12 and 13 are walls)
    # # The expert will avoid these, but the nominal MDP doesn't know about them yet.
    # true_constraints = {12, 13}
    
    # # 3. Generate Mock Expert Trajectories that avoid 12 and 13
    # # Trajectory format: list of (state, action)
    # mock_trajs = [
    #     [(0, 3), (1, 3), (2, 3), (3, 1), (8, 1), (13, 3), (14, 1), (19, 1), (24, None)], # Suboptimal, ignores constraints intentionally for mock
    #     [(5, 3), (6, 1), (11, 1), (16, 3), (17, 3), (18, 3), (19, 1), (24, None)],       # Avoids 12, 13
    #     [(0, 1), (5, 1), (10, 1), (15, 1), (20, 3), (21, 3), (22, 3), (23, 3), (24, None)] # Bottom edge path
    # ]
    
    print(f"True hidden constraints: {true_constraints}")
    print("Running MLCI to infer constraints...")
    
    # 4. Run MLCI
    inferred_constraints = run_mlci_inference(mdp, mock_trajs, d_kl_threshold=0.01)
    
    print("\nFinal Result:")
    print(f"Inferred Constraints: {inferred_constraints}")
    return inferred_constraints

#inferred_constraints= run_algo()  