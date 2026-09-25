"""GridWorld use case for MOCI-IRL. Defines an NxN grid with water (hard constraint),
grass, and rock terrain, and two ground-truth experts with different reward weights.
Generates expert demonstrations under the true constraint, builds a learner-only view
of the environment with the ground truth removed, and runs MOCI's EM algorithm
(MOCI_IRL.py) to jointly infer the hidden constraint set and each expert's reward
weights from the demonstrations alone. Also evaluates the inferred constraints against
ground truth and generates trajectory and learned-preference plots via
gridworld_Env.py."""
import copy
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from use_case.GridWorld import gridworld_Env as gw
except ModuleNotFoundError:
    import gridworld_Env as gw
import MOCI_IRL as moci

import copy


GRID_SIZE = 8
WATER = [12, 17, 38, 42, 43,50,53]
GRASS = [3, 7, 13, 29, 33, 34, 19, 39, 49]
ROCKS = [20, 6, 11, 21, 25, 26, 32, 40, 51, 52]



N_DEMOS_EXPERT1 = 20
N_DEMOS_EXPERT2 = 20

NUM_EXPERTS = 2
D_DKL = 0.05
LAMBDA_PENALTY = None
MAX_EM_ITERS = 10
SEARCH_OPTIONS = moci.SearchOptions.extended()

mdp = gw.CustomizableFeatureMDP(GRID_SIZE, WATER, GRASS, ROCKS)

w1 = np.array([0, 0, 0, 0])
w2 = np.array([0, 0, 0, 0])


def generate_expert_demos(seed=None):
    """Args: seed (optional RNG seed).
    Returns: (all_demos, resp), demos for both experts and their one-hot cluster labels."""
    if seed is not None:
        np.random.seed(seed)
    forbidden = set(WATER)
    z1 = moci.backward_pass(mdp, w1, forbidden)
    z2 = moci.backward_pass(mdp, w2, forbidden)

    def one_demo(w, z):
        """Args: w (reward weights), z (backward-pass value function).
        Returns: an explicit (state, action) demo trajectory."""
        states, actions = moci.sample_traj(mdp, w, z, constraints=forbidden, return_actions=True)
        return moci.to_explicit_demo(states, actions)

    all_demos = [one_demo(w1, z1) for _ in range(N_DEMOS_EXPERT1)] + [one_demo(w2, z2) for _ in range(N_DEMOS_EXPERT2)]

    resp = np.zeros((N_DEMOS_EXPERT1 + N_DEMOS_EXPERT2, 2))
    resp[:N_DEMOS_EXPERT1, 0] = 1
    resp[N_DEMOS_EXPERT1:, 1] = 1
    return all_demos, resp


def define_mdp_and_demos(visualize: bool = True, seed=None):
    """Args: visualize (save trajectory plot), seed (optional RNG seed).
    Returns: (w1, w2, WATER, mdp, all_demos, resp)."""
    all_demos, resp = generate_expert_demos(seed=seed)

    if visualize:
        gw.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        gw.plot_grid_setup(
            mdp,
            "Expert Trajectories (Lime=Grass Preference, Orange=Rock Preference)",
            [moci.demo_states(d) for d in all_demos],
            resp,
        )

    return w1, w2, WATER, mdp, all_demos, resp


def learner_view_of_mdp(env_mdp):
    """Args: env_mdp, the ground-truth MDP.
    Returns: a deep copy with the ground-truth water_states attribute removed."""
    view = copy.deepcopy(env_mdp)
    if hasattr(view, "water_states"):
        del view.water_states
    assert getattr(view, "action_feasible_mask", None) is None, "learner MDP must not carry constraints"
    assert not hasattr(view, "water_states")
    return view


def learn_from_demos(learner_mdp, demos, num_experts=None, d_dkl=None, max_em_iters=None,
                     lambda_penalty="config", options=None):
    """Args: learner_mdp, demos, and optional overrides (num_experts, d_dkl, max_em_iters,
    lambda_penalty, options) of the module-level defaults.
    Returns: (inferred_c, final_weights, final_priors) from moci.run_em_moci."""
    return moci.run_em_moci(
        learner_mdp,
        demos,
        K=NUM_EXPERTS if num_experts is None else num_experts,
        d_DKL=D_DKL if d_dkl is None else d_dkl,
        max_em_iters=MAX_EM_ITERS if max_em_iters is None else max_em_iters,
        lambda_penalty=LAMBDA_PENALTY if lambda_penalty == "config" else lambda_penalty,
        options=SEARCH_OPTIONS if options is None else options,
    )


def inferred_responsibilities(learner_mdp, demos, inferred_c, final_weights, final_priors):
    """Args: learner_mdp, demos, inferred_c, final_weights, final_priors.
    Returns: responsibility matrix with columns ordered [grass-preferring, rock-preferring]."""
    resp = moci.e_step(learner_mdp, demos, inferred_c, final_weights, np.array(final_priors, dtype=float))
    order = [0, 1] if final_weights[0][1] > final_weights[0][2] else [1, 0]
    return resp[:, order]


def plot_learned_preferences(final_weights, features=("Grass", "Rocks", "Water"), feature_idx=(1, 2, 3),
                             out_dir="Results/results_gridworld", name="joint_recovery_of_heterogeneous_preferences", show=True):
    """Args: final_weights, features/feature_idx to plot, out_dir/name for the saved figure, show.
    Draws a bar chart of learned reward weights per expert; saves <out_dir>/<name>.pdf and .png.
    Returns: path to the saved .png."""
    ink, ink2, grid, axis, surface = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#ffffff"
    colors = ("#1baf7a", "#eb6834")

    w = [np.asarray(x, dtype=float) for x in final_weights]
    lean = [x[1] - x[2] for x in w]
    order = [int(np.argmax(lean)), int(np.argmin(lean))]
    if order[0] == order[1]:
        order = [0, 1]
    idx = list(feature_idx)
    vals = [w[k][idx] / (np.linalg.norm(w[k][idx]) + 1e-8) for k in order]
    names = ("Expert 1 (grass-lover)", "Expert 2 (rock-lover)")

    x = np.arange(len(features))
    off, bw = 0.2, 0.34
    span = max(float(np.max(np.abs(vals))), 1e-3)

    with plt.rc_context({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10, "axes.linewidth": 0.8, "axes.edgecolor": axis, "axes.labelcolor": ink2,
        "xtick.color": ink2, "ytick.color": ink2, "text.color": ink, "pdf.fonttype": 42, "ps.fonttype": 42,
    }):
        fig, ax = plt.subplots(figsize=(6.6, 3.9), facecolor=surface)
        for j, (v, color) in enumerate(zip(vals, colors)):
            xs = x + (j - 0.5) * 2 * off
            ax.bar(xs, v, width=bw, color=color, edgecolor=surface, linewidth=0.8, zorder=3)
            for xi, yi in zip(xs, v):
                ax.text(xi, yi + np.sign(yi if yi else 1) * 0.02 * span, f"{yi:+.2f}", ha="center",
                        va="bottom" if yi >= 0 else "top", fontsize=8.5, color=ink2)
        ax.axhline(0, color=axis, linewidth=0.8, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(list(features))
        ax.set_ylabel("Learned reward weight")
        ax.set_ylim(min(0.0, min(v.min() for v in vals)) - 0.14 * span, max(0.0, max(v.max() for v in vals)) + 0.16 * span)
        ax.yaxis.grid(True, color=grid, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title("Joint Recovery of Heterogeneous Preferences", loc="left", pad=12, fontsize=11.5, fontweight="bold")
        ax.legend(
            handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=n) for c, n in zip(colors, names)],
            loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False,
            handlelength=1.0, handleheight=1.0, columnspacing=2.0,
        )
        fig.tight_layout(rect=(0, 0.03, 1, 1))

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
    return out / f"{name}.png"


def evaluate_against_ground_truth(inferred_c):
    """Args: inferred_c, the inferred constraint set.
    Returns: dict of precision/recall/FPR/TP/FP/FN against the true WATER constraint."""
    core_states = [s for s in range(mdp.num_states) if s not in (mdp.start_state, mdp.goal_state)]
    m = moci.detection_metrics({int(s) for s in inferred_c}, set(WATER), universe=core_states)
    return {
        "constraint_precision": m["precision"],
        "constraint_recall": m["tpr"],
        "false_positive_rate": m["fpr"],
        "true_positives": m["tp"],
        "false_positives": m["fp"],
        "false_negatives": m["fn"],
    }


def run_gridworld_moci_experiment(d_dkl=None, max_em_iters=None, visualize: bool = True,
                                  evaluate: bool = True, seed=None):
    """Args: d_dkl, max_em_iters, visualize, evaluate, seed (all optional).
    Runs the full pipeline: generate demos, learn, optionally plot and evaluate.
    Returns: (inferred_c, final_weights, final_priors)."""
    _, _, _, env_mdp, all_demos, resp = define_mdp_and_demos(visualize=False, seed=seed)

    learner_mdp = learner_view_of_mdp(env_mdp)
    inferred_c, final_weights, final_priors = learn_from_demos(
        learner_mdp, all_demos, d_dkl=d_dkl, max_em_iters=max_em_iters
    )

    if visualize:
        gw.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        gw.plot_grid_subfigures(
            mdp=learner_mdp,
            demos=[moci.demo_states(d) for d in all_demos],
            resp=resp,
            inf_c=inferred_c,
        )
        plot_learned_preferences(final_weights)

    if evaluate:
        metrics = evaluate_against_ground_truth(inferred_c)
        print(f"Constraint recovery vs ground truth: {metrics}")

    return inferred_c, final_weights, final_priors


if __name__ == "__main__":
    inferred_c, final_weights, final_priors = run_gridworld_moci_experiment()
    print(f"Ground Truth WATER tiles: {WATER}")
    print(f"Inferred Constraints: {sorted(list(inferred_c))}")
    print(f"Final Weights: {final_weights}")
    print(f"Final Priors: {final_priors}")
