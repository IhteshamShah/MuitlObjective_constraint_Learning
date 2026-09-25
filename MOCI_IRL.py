"""MOCI-IRL jointly infers per-cluster reward weights, and a set of
hard constraints (forbidden states or state-action pairs) from expert demonstrations. It
provides MaxEnt IRL primitives (backward_pass, trajectory sampling and log-likelihood), an
E-step/M-step EM loop (e_step, m_step_weights, m_step_constraints, run_em_moci) with a
SearchOptions configuration for joint vs. frozen-baseline search, path vs. local
likelihood, and pointwise vs. rule-level constraint generalization, a bootstrap ensemble
runner (run_moci_ensemble) for stability voting, and detection-metric utilities for
evaluating recovered constraints against ground truth.
"""
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.special import logsumexp


def _is_action_feasible(mdp, state_idx, action_idx):
    """Args: mdp, state_idx, action_idx. Returns True if the action is feasible in that state."""
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
    """Args: state_idx, action_idx, constraint_set. Returns True if the state or (state, action) pair is forbidden."""
    if state_idx in constraint_set:
        return True
    if action_idx is not None and (state_idx, action_idx) in constraint_set:
        return True
    return False


def _is_explicit(xi):
    """Args: xi (a demonstration). Returns True if it is in explicit (state, action) form."""
    return len(xi) > 0 and isinstance(xi[0], (tuple, list))


def _traj_states(xi):
    """Args: xi (a demonstration). Returns the list of visited states."""
    return [step[0] for step in xi] if _is_explicit(xi) else list(xi)


def _explicit_steps(xi):
    """Args: xi (an explicit demonstration). Returns the list of (s_t, a_t) for non-terminal steps."""
    return [(int(xi[i][0]), int(xi[i][1])) for i in range(len(xi) - 1)]


def demo_states(xi):
    """Args: xi (a demonstration, explicit or legacy). Returns the list of visited states."""
    return _traj_states(xi)


def to_explicit_demo(states, actions, terminal_action=-1):
    """Args: states, actions, terminal_action. Returns the explicit [(s, a), ...] demonstration."""
    if len(actions) != len(states) - 1:
        raise ValueError("need one action per transition (len(actions) == len(states) - 1)")
    return [(int(s), int(a)) for s, a in zip(states[:-1], actions)] + [(int(states[-1]), terminal_action)]


def resolve_lambda(spec, n_demos, d_DKL=0.0):
    """Args: spec (None, 'bic', 'aic', or a number), n_demos, d_DKL. Returns the resolved lambda penalty (float)."""
    if spec is None:
        return float(d_DKL) * n_demos
    if isinstance(spec, str):
        key = spec.lower()
        if key == "bic":
            return 0.5 * float(np.log(max(n_demos, 2)))
        if key == "aic":
            return 1.0
        raise ValueError(f"Unknown lambda_penalty '{spec}' (use None, 'bic', 'aic' or a number).")
    return float(spec)


def _passes_safety_filter(mdp, s, a, safety_filter):
    """Args: mdp, s, a, safety_filter (callable or set, or None). Returns True if (s, a) passes the filter."""
    if safety_filter is None:
        return True
    if callable(safety_filter):
        return bool(safety_filter(mdp, s, a))
    return (s, a) in safety_filter


@dataclass
class SearchOptions:
    """Configuration for the constraint search.

    Fields: mode, likelihood, weight_bound, refit_iters, candidate_epsilon,
    candidate_max_count, violation_noise, rank_by_discrepancy, generalize.
    """

    mode: str = "joint"
    likelihood: str = "path"
    weight_bound: Optional[float] = None
    refit_iters: int = 3
    candidate_epsilon: Optional[float] = None
    candidate_max_count: int = 0
    violation_noise: Optional[float] = None
    rank_by_discrepancy: bool = False
    generalize: Optional[str] = None

    @classmethod
    def plain(cls, **overrides):
        """Args: overrides (field overrides). Returns a SearchOptions with defaults for the plain algorithm."""
        return cls(**overrides)

    @classmethod
    def extended(cls, **overrides):
        """Args: overrides (field overrides). Returns a SearchOptions with defaults for the extended algorithm."""
        settings = dict(mode="frozen", likelihood="local", generalize="rule", refit_iters=3)
        settings.update(overrides)
        return cls(**settings)


def _validate_options(options):
    """Args: options (SearchOptions). Returns None; raises ValueError on invalid settings."""
    if options.mode not in ("joint", "frozen"):
        raise ValueError(f"SearchOptions.mode must be 'joint' or 'frozen', got {options.mode!r}")
    if options.likelihood not in ("path", "local"):
        raise ValueError(f"SearchOptions.likelihood must be 'path' or 'local', got {options.likelihood!r}")
    if options.generalize not in (None, "expand", "rule"):
        raise ValueError(f"SearchOptions.generalize must be None, 'expand' or 'rule', got {options.generalize!r}")
    if options.violation_noise is not None and not (0.0 < options.violation_noise < 1.0):
        raise ValueError("SearchOptions.violation_noise must lie in (0, 1)")
    allows_observed = options.candidate_max_count > 0 or options.candidate_epsilon is not None
    if allows_observed and options.violation_noise is None:
        raise ValueError(
            "candidate_epsilon / candidate_max_count admit demonstrated actions as candidates; "
            "set violation_noise so violating demonstrations are down-weighted instead of impossible."
        )


def _apply_options(mdp, options):
    """Args: mdp, options (SearchOptions). Sets mdp.likelihood_mode and mdp.violation_log_penalty; returns None."""
    mdp.likelihood_mode = options.likelihood
    eta = options.violation_noise
    mdp.violation_log_penalty = float(np.log(eta)) if eta else -np.inf


def _feasible_matrix(mdp):
    """Args: mdp. Returns an S x A boolean ndarray of feasible actions."""
    if getattr(mdp, "action_feasible_mask", None) is None:
        return np.ones((mdp.num_states, mdp.num_actions), dtype=bool)
    return np.array(
        [[_is_action_feasible(mdp, s, a) for a in range(mdp.num_actions)] for s in range(mdp.num_states)],
        dtype=bool,
    )


def _allowed_matrix(mdp, obstacles):
    """Args: mdp, obstacles (constraint set). Returns an S x A boolean ndarray of allowed actions."""
    allowed = _feasible_matrix(mdp)
    for c in obstacles:
        if isinstance(c, tuple):
            allowed[c[0], c[1]] = False
    return allowed


def backward_pass(mdp, weights, obstacles):
    """Args: mdp, weights, obstacles (constraint set). Returns the log partition matrix logZ(C, w)."""
    S, H = mdp.num_states, mdp.horizon
    logZ = np.full((S, H + 1), -np.inf, dtype=float)
    rewards = np.dot(mdp.feature_map, weights)
    trans = np.asarray(mdp.transitions)
    goal = mdp.goal_state

    state_constraints = [int(c) for c in obstacles if isinstance(c, (int, np.integer))]
    allowed = _allowed_matrix(mdp, obstacles)

    logZ[:, H] = 0.0
    logZ[goal, H] = 10.0
    if state_constraints:
        logZ[state_constraints, :] = -np.inf

    with np.errstate(divide="ignore", invalid="ignore"):
        for t in range(H - 1, -1, -1):
            terms = np.where(allowed, rewards + logZ[trans, t + 1], -np.inf)
            col = logsumexp(terms, axis=1)
            col[goal] = 10.0 + logZ[goal, t + 1]
            if state_constraints:
                col[state_constraints] = -np.inf
            logZ[:, t] = col

    return logZ


def calculate_trajectory_prob(mdp, xi, C, w_k, logZ_matrix):
    """Args: mdp, xi (demonstration), C (constraint set), w_k (weights), logZ_matrix. Returns the log probability of xi (-inf if impossible)."""
    states = _traj_states(xi)
    for state in states:
        if state in C:
            return -np.inf

    rewards = np.dot(mdp.feature_map, w_k)
    path_reward = 0.0
    if _is_explicit(xi):
        vpen = getattr(mdp, "violation_log_penalty", -np.inf)
        local = getattr(mdp, "likelihood_mode", "path") == "local"
        H = logZ_matrix.shape[1] - 1
        for j, (s, a) in enumerate(_explicit_steps(xi)):
            if _is_forbidden_constraint(s, a, C) or not _is_action_feasible(mdp, s, a):
                if not np.isfinite(vpen):
                    return -np.inf
                path_reward += vpen
                continue
            if local:
                nxt = logZ_matrix[mdp.transitions[s, a], min(j + 1, H)]
                cur = logZ_matrix[s, min(j, H)]
                if not (np.isfinite(nxt) and np.isfinite(cur)):
                    return -np.inf
                path_reward += rewards[s, a] + nxt - cur
            else:
                path_reward += rewards[s, a]
        if local:
            return path_reward
    else:
        unmatched_log_prob = np.log(1e-12)
        for i in range(len(states) - 1):
            s, sn = states[i], states[i+1]
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
    if states[-1] == mdp.goal_state:
        path_reward += 10.0

    return path_reward - log_z_0

def sample_traj(mdp, weights, logZ, constraints=None, return_actions=False):
    """Args: mdp, weights, logZ, constraints, return_actions. Returns a sampled trajectory (list of states), or (states, actions) if return_actions."""
    curr, traj, acts = mdp.start_state, [mdp.start_state], []
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
        acts.append(int(action_idx))
    return (traj, acts) if return_actions else traj


def reachable_states(mdp):
    """Args: mdp. Returns the set of states reachable from the start state under T(s, a)."""
    seen = {int(mdp.start_state)}
    frontier = [int(mdp.start_state)]
    while frontier:
        s = frontier.pop()
        for a in range(mdp.num_actions):
            if not _is_action_feasible(mdp, s, a):
                continue
            n = int(mdp.transitions[s, a])
            if n not in seen:
                seen.add(n)
                frontier.append(n)
    return seen


def empirical_action_stats(mdp, D):
    """Args: mdp, D (demonstrations). Returns (sa_counts, s_counts), Counters of (s, a) and s occurrences."""
    sa_counts, s_counts = Counter(), Counter()
    for xi in D:
        if _is_explicit(xi):
            for s, a in _explicit_steps(xi):
                sa_counts[(s, a)] += 1
                s_counts[s] += 1
        else:
            for i in range(len(xi) - 1):
                s, sn = xi[i], xi[i + 1]
                s_counts[s] += 1
                for a in range(mdp.num_actions):
                    if _is_action_feasible(mdp, s, a) and mdp.transitions[s, a] == sn:
                        sa_counts[(s, a)] += 1
    return sa_counts, s_counts


def model_policy(mdp, weights, priors, C):
    """Args: mdp, weights (per cluster), priors, C (constraint set). Returns the S x A MaxEnt policy pi_model(a | s)."""
    S, A, H = mdp.num_states, mdp.num_actions, mdp.horizon
    trans = np.asarray(mdp.transitions)
    allowed = _allowed_matrix(mdp, C)
    pri = np.clip(np.asarray(priors, dtype=float), 1e-12, None)
    pri = pri / pri.sum()

    pi = np.zeros((S, A))
    for k, w in enumerate(weights):
        logZ = backward_pass(mdp, w, C)
        r = np.dot(mdp.feature_map, w)
        acc, n = np.zeros((S, A)), np.zeros(S)
        with np.errstate(divide="ignore", invalid="ignore"):
            for t in range(H):
                terms = np.where(allowed, r + logZ[trans, t + 1], -np.inf)
                lse = logsumexp(terms, axis=1)
                ok = np.isfinite(lse)
                p = np.zeros((S, A))
                p[ok] = np.exp(terms[ok] - lse[ok, None])
                acc += p
                n += ok
        pi += pri[k] * acc / np.maximum(n, 1)[:, None]
    return pi


def identify_candidates(mdp, D, safety_filter=None, epsilon=None, max_count=0):
    """Args: mdp, D (demonstrations), safety_filter, epsilon, max_count. Returns the list of candidate constraints (states or (s, a) pairs)."""
    if getattr(mdp, "constraint_mode", "state") == "state_action":
        sa_counts, s_counts = empirical_action_stats(mdp, D)

        candidate_pairs = []
        for s in sorted(s_counts):
            if s == mdp.goal_state or s == mdp.start_state:
                continue
            for a in range(mdp.num_actions):
                if not _is_action_feasible(mdp, s, a):
                    continue
                n = sa_counts.get((s, a), 0)
                rare = n <= max_count or (epsilon is not None and n / s_counts[s] < epsilon)
                if not rare:
                    continue
                if not _passes_safety_filter(mdp, s, a, safety_filter):
                    continue
                candidate_pairs.append((s, a))

        return candidate_pairs

    visited = set(s for xi in D for s in _traj_states(xi))
    return [s for s in range(mdp.num_states) if s not in visited and s != mdp.goal_state]


def _pair_signature(mdp, s, a, feature_matrix=None):
    """Args: mdp, s, a, feature_matrix. Returns the signature of (s, a) (from mdp.constraint_signature or rounded feature vector)."""
    sig = getattr(mdp, "constraint_signature", None)
    if feature_matrix is None and sig is not None:
        return sig.get((s, a))
    fm = mdp.feature_map if feature_matrix is None else feature_matrix
    return tuple(np.round(np.asarray(fm[s, a], dtype=float), 6).tolist())


def _unit_key(mdp, c, feature_matrix=None):
    """Args: mdp, c (a state or (s, a) pair), feature_matrix. Returns the grouping key for candidates sharing a rule."""
    sig_map = getattr(mdp, "constraint_signature", None)
    if isinstance(c, tuple):
        s, a = c
        sig = _pair_signature(mdp, s, a, feature_matrix)
        return (a, sig) if sig is not None else (a, ("__single__", s))
    s = int(c)
    sig = sig_map.get(s) if (feature_matrix is None and sig_map is not None) else None
    if sig is None:
        fm = mdp.feature_map if feature_matrix is None else feature_matrix
        sig = tuple(np.round(np.asarray(fm[s], dtype=float).mean(axis=0), 6).tolist())
    return ("state", sig)


def _signature_groups(mdp, pool, feature_matrix=None):
    """Args: mdp, pool (candidates), feature_matrix. Returns a dict mapping rule key to a tuple of candidates."""
    groups = {}
    for c in pool:
        groups.setdefault(_unit_key(mdp, c, feature_matrix), []).append(c)
    return {k: tuple(v) for k, v in groups.items()}


def expand_constraint_by_features(mdp, confirmed_constraint, pool=None, feature_matrix=None):
    """Args: mdp, confirmed_constraint (a state or (s, a) pair), pool, feature_matrix. Returns the list of candidates sharing its signature."""
    target = _unit_key(mdp, confirmed_constraint, feature_matrix)
    if pool is None:
        core = [s for s in range(mdp.num_states) if s not in (mdp.start_state, mdp.goal_state)]
        if isinstance(confirmed_constraint, tuple):
            a_t = confirmed_constraint[1]
            pool = [(s, a_t) for s in core if _is_action_feasible(mdp, s, a_t)]
        else:
            pool = core
    expanded = [c for c in pool if _unit_key(mdp, c, feature_matrix) == target]
    if confirmed_constraint not in expanded:
        expanded.append(confirmed_constraint)
    return expanded


def e_step(mdp, D, C_hat, weights, priors):
    """Args: mdp, D (demonstrations), C_hat, weights (per cluster), priors. Returns the responsibilities matrix gamma_{i,k}."""
    num_demos = len(D)
    K = len(weights)
    log_gamma = np.zeros((num_demos, K))
    safe_priors = np.clip(priors, 1e-12, None)
    safe_priors /= safe_priors.sum()

    Zs = [backward_pass(mdp, weights[k], C_hat) for k in range(K)]

    for i, xi_i in enumerate(D):
        for k in range(K):
            log_prob = calculate_trajectory_prob(mdp, xi_i, C_hat, weights[k], Zs[k])
            log_gamma[i, k] = np.log(safe_priors[k]) + log_prob

        finite = np.isfinite(log_gamma[i, :])
        if not np.any(finite):
            log_gamma[i, :] = -np.inf
            continue
        log_gamma[i, finite] -= logsumexp(log_gamma[i, finite])

    return np.exp(log_gamma)

def m_step_weights(mdp, D, C_hat, weights, responsibilities, lr=0.1, steps=5, weight_bound=None):
    """Args: mdp, D, C_hat, weights, responsibilities, lr, steps, weight_bound. Returns the updated per-cluster weights (list of ndarrays)."""
    K = len(weights)
    new_weights = [np.copy(w) for w in weights]
    vpen = getattr(mdp, "violation_log_penalty", -np.inf)

    for k in range(K):
        for _ in range(steps):
            Z = backward_pass(mdp, new_weights[k], C_hat)

            exp_counts = np.zeros(mdp.num_features)
            num_samples = 100
            for _ in range(num_samples):
                sample, sample_acts = sample_traj(mdp, new_weights[k], Z, C_hat, return_actions=True)
                for s, a in zip(sample[:-1], sample_acts):
                    exp_counts += mdp.feature_map[s, a]
            exp_counts /= num_samples

            grad = np.zeros(mdp.num_features)

            for i, xi_i in enumerate(D):
                if responsibilities[i, k] < 1e-3: continue

                emp_counts = np.zeros(mdp.num_features)
                if _is_explicit(xi_i):
                    demo_steps = _explicit_steps(xi_i)
                    violating = [_is_forbidden_constraint(s, a, C_hat) for s, a in demo_steps]
                    if any(violating) and not np.isfinite(vpen):
                        continue
                    for (s, a), bad in zip(demo_steps, violating):
                        if not bad:
                            emp_counts += mdp.feature_map[s, a]
                else:
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

                grad += responsibilities[i, k] * (emp_counts - exp_counts)

            new_weights[k] += lr * grad / len(D)
            if weight_bound is not None:
                np.clip(new_weights[k], -weight_bound, weight_bound, out=new_weights[k])

    return new_weights

def calculate_joint_log_likelihood (mdp, D, C, weights, priors):
    """Args: mdp, D, C, weights, priors. Returns the average log-likelihood L_avg(C, weights, priors)."""
    total_log_L = 0
    Zs = [backward_pass(mdp, weights[k], C) for k in range(len(weights))]
    safe_priors = np.clip(priors, 1e-12, None)
    safe_priors /= safe_priors.sum()

    for xi in D:
        log_probs = []
        for k in range(len(weights)):
            log_prob = calculate_trajectory_prob(mdp, xi, C, weights[k], Zs[k])
            log_probs.append(np.log(safe_priors[k]) + log_prob)
        total_log_L += logsumexp(log_probs)

    return total_log_L / len(D)

def calculate_joint_log_likelihood_old (mdp, D, C, weights, priors):
    """Args: mdp, D, C, weights, priors. Returns the total (non-averaged) log-likelihood."""
    total_log_L = 0
    Zs = [backward_pass(mdp, weights[k], C) for k in range(len(weights))]
    safe_priors = np.clip(priors, 1e-12, None)
    safe_priors /= safe_priors.sum()

    for xi in D:
        log_probs = []
        for k in range(len(weights)):
            log_prob = calculate_trajectory_prob(mdp, xi, C, weights[k], Zs[k])
            log_probs.append(np.log(safe_priors[k]) + log_prob)
        total_log_L += logsumexp(log_probs)
    return total_log_L

def m_step_constraints(mdp, D, C_hat, weights, priors, d_DKL, candidate_subset_size=None,
                       lambda_penalty=None, safety_filter=None, options=None, diagnostics=None):
    """Args: mdp, D, C_hat, weights, priors, d_DKL, candidate_subset_size, lambda_penalty, safety_filter, options (SearchOptions), diagnostics (dict). Returns the updated constraint set."""
    options = options or SearchOptions()
    _validate_options(options)
    _apply_options(mdp, options)
    n_demos = len(D)
    lam = resolve_lambda(lambda_penalty, n_demos, d_DKL)

    pool = identify_candidates(
        mdp, D, safety_filter=safety_filter,
        epsilon=options.candidate_epsilon, max_count=options.candidate_max_count,
    )
    groups = _signature_groups(mdp, pool) if options.generalize else {}
    if options.generalize == "rule":
        candidates = [tuple(g) for g in groups.values()]
    else:
        candidates = [(c,) for c in pool]
    group_of = {p: g for g in groups.values() for p in g}

    rank = bool(options.rank_by_discrepancy) and len(pool) > 0 and isinstance(pool[0], tuple)
    if options.rank_by_discrepancy and not rank and len(pool) > 0:
        print("  (rank_by_discrepancy applies to state-action constraints only; ignored for state constraints)")
    sa_counts, s_counts = empirical_action_stats(mdp, D) if rank else (None, None)
    floor = diagnostics.get("shadow_floor") if diagnostics is not None else None
    max_shadow = diagnostics.get("max_shadow_rounds", 80) if diagnostics is not None else 0
    trace = diagnostics.setdefault("trace", []) if diagnostics is not None else None
    if diagnostics is not None:
        diagnostics["n_candidates"] = len(pool)
        diagnostics["n_candidate_units"] = len(candidates)

    current_L = calculate_joint_log_likelihood(mdp, D, C_hat, weights, priors)
    print(f"  M-step C: {len(pool)} candidates ({len(candidates)} units), lambda={lam:.4f}")

    work_C, shadow, shadow_rounds = C_hat, False, 0
    while candidates:
        if rank:
            pi = model_policy(mdp, weights, priors, work_C)

            def risk(unit):
                """Args: unit (candidate pairs). Returns the mean model-minus-expert action-probability risk."""
                return float(np.mean([
                    pi[s, a] - sa_counts.get((s, a), 0) / max(s_counts.get(s, 0), 1)
                    for (s, a) in unit
                ]))

            candidates.sort(key=lambda unit: -risk(unit))

        if candidate_subset_size is None or candidate_subset_size >= len(candidates):
            subset = list(candidates)
        elif candidate_subset_size <= 0:
            subset = list(candidates)
        else:
            subset = list(candidates)[: int(candidate_subset_size)]

        best_c, best_L = None, -np.inf
        for unit in subset:
            test_L = calculate_joint_log_likelihood(mdp, D, work_C.union(unit), weights, priors)
            if test_L > best_L:
                best_L = test_L
                best_c = unit

        if best_c is None:
            break

        delta_logL = (best_L - current_L) * n_demos

        if not shadow:
            if not (delta_logL - lam > 0.0):
                if floor is None or not (delta_logL > floor):
                    break
                shadow, work_C = True, set(C_hat)
        else:
            shadow_rounds += 1
            if not (delta_logL > floor):
                break
            if shadow_rounds > max_shadow:
                diagnostics["shadow_capped"] = True
                break

        new_pairs = set(best_c)
        if options.generalize == "expand":
            new_pairs = set(group_of.get(best_c[0], best_c))
        work_C.update(new_pairs)
        if trace is not None:
            trace.append({
                "pairs": sorted(new_pairs),
                "delta_logL": float(delta_logL),
                "accepted": not shadow,
            })

        candidates = [u for u in candidates if not set(u) <= work_C]
        current_L = best_L if new_pairs == set(best_c) else calculate_joint_log_likelihood(
            mdp, D, work_C, weights, priors
        )

    return C_hat

def run_em_moci(mdp, D, K, d_DKL, max_em_iters, candidate_subset_size=None,
                lambda_penalty=None, safety_filter=None, options=None, diagnostics=None):
    """Args: mdp, D (demonstrations), K (clusters), d_DKL, max_em_iters, candidate_subset_size, lambda_penalty, safety_filter, options (SearchOptions), diagnostics (dict). Returns (C_hat, weights, priors)."""
    options = options or SearchOptions()
    _validate_options(options)
    _apply_options(mdp, options)

    C_hat = set()
    num_features = mdp.num_features
    weights = [np.random.randn(num_features) * 0.1 for _ in range(K)]
    priors = np.full(K, 1.0 / K)

    def refit_weights(C, weights, priors, iters):
        """Args: C (constraint set), weights, priors, iters. Returns the (weights, priors) refitted over `iters` EM steps."""
        for _ in range(iters):
            responsibilities = e_step(mdp, D, C, weights, priors)
            priors = np.clip(np.mean(responsibilities, axis=0), 1e-12, None)
            priors /= priors.sum()
            weights = m_step_weights(mdp, D, C, weights, responsibilities, weight_bound=options.weight_bound)
        return weights, priors

    if options.mode == "frozen":
        print("--- Baseline fit (C = {}) ---")
        weights, priors = refit_weights(set(), weights, priors, max_em_iters)
        responsibilities = e_step(mdp, D, set(), weights, priors)
        priors = np.clip(np.mean(responsibilities, axis=0), 1e-12, None)
        priors /= priors.sum()
        if diagnostics is not None:
            diagnostics["baseline_weights"] = [np.copy(w) for w in weights]
            diagnostics["baseline_priors"] = np.copy(priors)

        print("--- Constraint search (weights frozen) ---")
        C_hat = m_step_constraints(
            mdp, D, C_hat, weights, priors, d_DKL,
            candidate_subset_size=candidate_subset_size,
            lambda_penalty=lambda_penalty,
            safety_filter=safety_filter,
            options=options,
            diagnostics=diagnostics,
        )
        print(f"Inferred Constraints ({len(C_hat)}): {sorted(list(C_hat))}")

        if options.refit_iters > 0:
            print("--- Re-fit weights under final C ---")
            weights, priors = refit_weights(C_hat, weights, priors, options.refit_iters)
        return C_hat, weights, priors

    for em_iter in range(max_em_iters):
        print(f"--- EM Iteration {em_iter + 1} ---")

        responsibilities = e_step(mdp, D, C_hat, weights, priors)

        priors = np.mean(responsibilities, axis=0)
        priors = np.clip(priors, 1e-12, None)
        priors /= priors.sum()

        weights = m_step_weights(mdp, D, C_hat, weights, responsibilities, weight_bound=options.weight_bound)

        C_hat = m_step_constraints(
            mdp, D, C_hat, weights, priors, d_DKL,
            candidate_subset_size=candidate_subset_size,
            lambda_penalty=lambda_penalty,
            safety_filter=safety_filter,
            options=options,
            diagnostics=diagnostics if em_iter == max_em_iters - 1 else None,
        )

        print(f"Current Inferred Constraints: {sorted(list(C_hat))}")

    if diagnostics is not None:
        diagnostics["baseline_weights"] = [np.copy(w) for w in weights]
        diagnostics["baseline_priors"] = np.copy(priors)
    return C_hat, weights, priors


@dataclass
class EnsembleResult:
    """Outcome of run_moci_ensemble.

    Fields: restarts (per-restart record dicts), votes (canonical constraint ->
    restart count), n_restarts, best (restart with highest log-likelihood).
    """

    restarts: list
    votes: dict
    n_restarts: int
    best: dict

    def probability(self, key):
        """Args: key (a canonical constraint). Returns the share of restarts that learned it (float)."""
        return self.votes.get(key, 0) / float(self.n_restarts)


def run_moci_ensemble(make_problem, K, n_restarts, seed_base=0, *, d_DKL=0.05, max_em_iters=10,
                      candidate_subset_size=None, lambda_penalty=None, options=None,
                      safety_filter_fn=None, canonicalize=None, diagnostics_factory=None):
    """Args: make_problem, K, n_restarts, seed_base, d_DKL, max_em_iters, candidate_subset_size, lambda_penalty, options, safety_filter_fn, canonicalize, diagnostics_factory. Returns an EnsembleResult."""
    options = options or SearchOptions()
    restarts, votes, best = [], Counter(), None

    for idx in range(n_restarts):
        seed = seed_base + idx
        rng = np.random.default_rng(seed)
        mdp, demos = make_problem(idx, rng)
        np.random.seed(seed)
        diag = diagnostics_factory(idx, mdp) if diagnostics_factory is not None else None

        constraints, weights, priors = run_em_moci(
            mdp, demos, K, d_DKL, max_em_iters,
            candidate_subset_size=candidate_subset_size,
            lambda_penalty=lambda_penalty,
            safety_filter=safety_filter_fn(mdp) if safety_filter_fn is not None else None,
            options=options,
            diagnostics=diag,
        )
        log_like = float(calculate_joint_log_likelihood(mdp, demos, constraints, weights, priors))
        canonical = set(canonicalize(mdp, constraints)) if canonicalize is not None else set(constraints)
        votes.update(canonical)

        record = {
            "restart": idx, "seed": seed, "mdp": mdp, "demos": demos, "constraints": constraints,
            "canonical": canonical, "weights": weights, "priors": priors, "log_like": log_like,
            "diagnostics": diag,
        }
        restarts.append(record)
        if best is None or log_like > best["log_like"]:
            best = record

    return EnsembleResult(restarts=restarts, votes=dict(votes), n_restarts=n_restarts, best=best)


def detection_metrics(predicted, truth, universe=None):
    """Args: predicted, truth (constraint sets), universe (all candidates, optional). Returns a dict of tp/fp/fn/tn/tpr/fpr/precision."""
    predicted, truth = set(predicted), set(truth)
    tp, fp, fn = len(predicted & truth), len(predicted - truth), len(truth - predicted)
    tn = len(set(universe) - predicted - truth) if universe is not None else None
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "tpr": tp / (tp + fn) if (tp + fn) else 0.0,
        "fpr": fp / (fp + tn) if (tn is not None and (fp + tn)) else (None if tn is None else 0.0),
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
    }
