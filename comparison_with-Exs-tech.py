import numpy as np
import copy

def calculate_kl_divergence(expert_trajs, mdp):
    """
    Calculates D_KL between the expert's empirical distribution 
    and the MDP's expected feature accrual (state visitation).
    """
    # 1. Calculate Empirical Distribution (p)
    total_states = mdp.num_states
    empirical_counts = np.zeros(total_states)
    total_visits = 0
    for traj in expert_trajs:
        for state, action in traj:
            empirical_counts[state] += 1
            total_visits += 1
    p = empirical_counts / (total_visits + 1e-9)

    # 2. Calculate MDP Expected Distribution (q)
    # This assumes your MDP has a method to get state visitation frequencies
    # For Maximum Entropy IRL, this is the State Expected Visitation Frequency (SVF)
    q = mdp.compute_state_visitation_freq() 
    
    # 3. Compute KL Divergence
    # We add a small epsilon to avoid log(0)
    kl = np.sum(p * np.log((p + 1e-9) / (q + 1e-9)))
    return kl

def run_mlci_inference(nominal_mdp, expert_trajs, d_kl_threshold=0.1, max_constraints=20):
    """
    Python reproduction of RunConstraintInference.m logic.
    """
    estimated_constraints = set()
    total_states = nominal_mdp.num_states
    
    # Initialize KL history
    current_mdp = copy.deepcopy(nominal_mdp)
    kl_history = [calculate_kl_divergence(expert_trajs, current_mdp)]
    
    print(f"Initial KL Divergence: {kl_history[0]:.4f}")

    # Identify "Unaccrued Features" (States never visited by the expert)
    visited_states = set()
    for traj in expert_trajs:
        for state, _ in traj:
            visited_states.add(state)
    
    candidate_constraints = [s for s in range(total_states) if s not in visited_states]

    for i in range(max_constraints):
        best_candidate = None
        best_kl_drop = -np.inf
        temp_best_mdp = None

        # MLCI Heuristic: Find the constraint that improves likelihood (reduces KL) the most
        # In the original MATLAB code, this is done via the 'max coverage' of unaccrued features
        for candidate in candidate_constraints:
            if candidate in estimated_constraints:
                continue
            
            # Create a temporary MDP with this candidate as a hard constraint
            test_constraints = estimated_constraints | {candidate}
            test_mdp = copy.deepcopy(nominal_mdp)
            test_mdp.set_constraints(test_constraints) # Assuming your MDP class supports this
            
            # Re-run Backward pass to update policy under new constraints
            # This corresponds to 'BackwardForward' in the MATLAB code
            test_mdp.update_policy() 
            
            new_kl = calculate_kl_divergence(expert_trajs, test_mdp)
            kl_drop = kl_history[-1] - new_kl
            
            if kl_drop > best_kl_drop:
                best_kl_drop = kl_drop
                best_candidate = candidate
                temp_best_mdp = test_mdp

        # Check if we found a beneficial constraint
        if best_candidate is None or best_kl_drop < 0:
            print("No more beneficial constraints found.")
            break

        # Apply the best candidate
        estimated_constraints.add(best_candidate)
        kl_history.append(calculate_kl_divergence(expert_trajs, temp_best_mdp))
        current_mdp = temp_best_mdp

        print(f"Iteration {i+1}: Added state {best_candidate} as constraint. KL: {kl_history[-1]:.4f}")

        # Stopping condition based on KL change (d_kl_threshold)
        if abs(kl_history[-2] - kl_history[-1]) < d_kl_threshold:
            print("Stopping due to KL convergence.")
            # Remove the last added constraint as per MATLAB logic (low impact)
            estimated_constraints.remove(best_candidate)
            break

    return estimated_constraints