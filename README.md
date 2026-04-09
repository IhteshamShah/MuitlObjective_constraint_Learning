# MuitlObjective_constraint_Learning
"""
Overview of the main components of the MLCI framework:

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