"""Runs three GridWorld ablation studies for MOCI. The oracle-reward ablation
compares MOCI, MLCI, and ICRL under a generic reward and two oracle
terrain-reward conditions matching each pooled expert. The EM-iterations
ablation sweeps MOCI's EM budget and measures constraint detection against
reward/cluster recovery. The K-ablation fits MOCI at K=1/2/3 over three
pooled experts (Grass, Rock, Sand) to measure how mixture capacity affects
log-likelihood, constraint recovery, and cluster metrics. Each ablation
includes trial runners, summarization, printing, and CSV/LaTeX/plot saving
helpers, reusing shared GridWorld setup and plotting utilities from
GW_comparison_main.py.
"""

import contextlib
import csv
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import MOCI_IRL as moci
import Additional_Experiments.GW_comparison_main as gwc
import Additional_Experiments.MLCI_existingWork as MLCI_exs

_ICRL_PATH = Path(__file__).resolve().parent / "ICLR.existiing.py"
_spec = importlib.util.spec_from_file_location("icrl_existing_ablation", _ICRL_PATH)
ICRL_exs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ICRL_exs)


SEED = 0
STEP_COST = -1.0
GOAL_REWARD = 50.0
MLCI_D_KL_THRESHOLD = 0.01


def oracle_state_reward(mdp, w, step_cost=STEP_COST, goal_reward=GOAL_REWARD):
    """Args: mdp, preference weights w, step_cost, goal_reward.
    Returns: per-state reward array."""
    terrain_reward = mdp.feature_map[:, 0, :] @ w
    reward = np.full(mdp.num_states, step_cost, dtype=float) + terrain_reward
    reward[mdp.goal_state] = goal_reward
    return reward


def run_mlci_with_reward(demos_mlci, grid_size, reward, d_kl_threshold=MLCI_D_KL_THRESHOLD):
    """Args: demos_mlci, grid_size, reward, d_kl_threshold.
    Returns: MLCI inferred constraint set."""
    mdp_local = MLCI_exs.SimpleGridMDP(grid_size)
    mdp_local.rewards = np.asarray(reward, dtype=float)
    return MLCI_exs.run_mlci_inference(mdp_local, demos_mlci, d_kl_threshold=d_kl_threshold)


def run_ablation(seed=SEED):
    """Args: seed.
    Returns: (results, water, n_states)."""
    np.random.seed(seed)
    with gwc._quiet():
        grid_size, w1, w2, water, shared_mdp, demos_mlci, demos_moci, resp = MLCI_exs.define_mdp_and_demos(
            gwc.mdp, gwc.WATER, gwc.W1, gwc.W2, gwc.N_DEMOS_EXPERT1, gwc.N_DEMOS_EXPERT2
        )

        t0 = time.time()
        inferred_moci, _, _ = moci.run_em_moci(shared_mdp, demos_moci, K=2, d_DKL=0.1, max_em_iters=1)
        moci_time = time.time() - t0

        reward_conditions = {
            "generic": ICRL_exs.make_known_reward(shared_mdp, step_cost=STEP_COST, goal_reward=GOAL_REWARD),
            "oracle_w1_grass": oracle_state_reward(shared_mdp, w1),
            "oracle_w2_rock": oracle_state_reward(shared_mdp, w2),
        }

        ablation_rows = {}
        for cond_name, reward in reward_conditions.items():
            t0 = time.time()
            inferred_mlci = run_mlci_with_reward(demos_mlci, grid_size, reward)
            mlci_time = time.time() - t0

            t0 = time.time()
            inferred_icrl, _, _ = ICRL_exs.run_icrl(shared_mdp, demos_mlci, reward, seed=seed)
            icrl_time = time.time() - t0

            ablation_rows[cond_name] = {
                "MLCI": (inferred_mlci, mlci_time),
                "ICRL": (inferred_icrl, icrl_time),
            }

    n_states = shared_mdp.num_states
    results = {
        "MOCI": {
            "n/a (joint)": (
                inferred_moci, moci_time,
                MLCI_exs.calculate_cmse(water, inferred_moci, n_states),
                *gwc._confusion_counts(water, inferred_moci, n_states),
            )
        }
    }
    for method in ("MLCI", "ICRL"):
        results[method] = {}
        for cond_name, cond_rows in ablation_rows.items():
            inferred, runtime_s = cond_rows[method]
            results[method][cond_name] = (
                inferred, runtime_s,
                MLCI_exs.calculate_cmse(water, inferred, n_states),
                *gwc._confusion_counts(water, inferred, n_states),
            )

    return results, water, n_states


def print_results(results: dict, water, n_states: int) -> None:
    """Args: results, water, n_states.
    Returns: None."""
    print(f"8x8 GridWorld (shared with GW_comparison_main.py), true WATER constraint: {sorted(water)}\n")

    header = f"{'Approach':<10}{'Reward condition':<24}{'Run-time (s)':>14}{'MSE':>9}{'TP':>5}{'FP':>5}{'FN':>5}{'TN':>5}{'Precision':>11}{'Recall':>9}"
    print(header)
    print("-" * len(header))
    for method, conditions in results.items():
        for cond_name, (inferred, runtime_s, mse, tp, fp, fn, tn, precision, recall) in conditions.items():
            print(
                f"{method:<10}{cond_name:<24}{runtime_s:>14.4f}{mse:>9.4f}"
                f"{tp:>5d}{fp:>5d}{fn:>5d}{tn:>5d}{precision:>11.4f}{recall:>9.4f}"
            )
        print()

    print("Inferred constraints by condition:")
    for method, conditions in results.items():
        for cond_name, (inferred, *_rest) in conditions.items():
            print(f"  {method:<6}[{cond_name}]: {sorted(list(inferred))}")


EM_ITERS_VALUES = (1, 3, 6, 9, 12, 15)
EM_ITERS_RESULTS_DIR = ROOT / "Results" / "results_additional_exp" / "em_iters_ablation"


def run_moci_only_trial(seed, moci_em_iters, moci_d_dkl=0.1):
    """Args: seed, moci_em_iters, moci_d_dkl.
    Returns: dict of per-seed metrics."""
    with gwc._quiet():
        np.random.seed(seed)
        grid_size, w1, w2, water, shared_mdp, demos_mlci, demos_moci, resp = MLCI_exs.define_mdp_and_demos(
            gwc.mdp, gwc.WATER, gwc.W1, gwc.W2, gwc.N_DEMOS_EXPERT1, gwc.N_DEMOS_EXPERT2
        )
        t0 = time.time()
        inferred_moci, final_weights, final_priors = moci.run_em_moci(
            shared_mdp, demos_moci, K=2, d_DKL=moci_d_dkl, max_em_iters=moci_em_iters
        )
        runtime_s = time.time() - t0

        responsibilities = moci.e_step(shared_mdp, demos_moci, inferred_moci, final_weights, np.array(final_priors))
        predicted_labels = np.argmax(responsibilities, axis=1)
        true_labels = np.array([0] * gwc.N_DEMOS_EXPERT1 + [1] * gwc.N_DEMOS_EXPERT2)
        cluster_ari = adjusted_rand_score(true_labels, predicted_labels)
        sim_w1, sim_w2 = gwc.match_weights_to_experts(final_weights, w1, w2)

    n_states = shared_mdp.num_states
    mse = MLCI_exs.calculate_cmse(water, inferred_moci, n_states)
    tp, fp, fn, tn, precision, recall = gwc._confusion_counts(water, inferred_moci, n_states)
    f1, mcc = gwc.calculate_f1_mcc(tp, fp, fn, tn)
    return {
        "seed": seed, "runtime_sec": runtime_s, "mse": mse, "precision": precision, "recall": recall,
        "f1": f1, "mcc": mcc, "weight_cosine_sim_expert1_grass": sim_w1,
        "weight_cosine_sim_expert2_rock": sim_w2, "cluster_ari": cluster_ari,
    }


def run_em_iters_ablation(em_iters_values=EM_ITERS_VALUES, seeds=None, moci_d_dkl=0.1):
    """Args: em_iters_values, seeds, moci_d_dkl.
    Returns: dict mapping em_iters to list of per-seed trial results."""
    seeds = seeds if seeds is not None else gwc.SEEDS
    return {
        em_iters: [run_moci_only_trial(seed, em_iters, moci_d_dkl) for seed in seeds]
        for em_iters in em_iters_values
    }


def summarize_em_iters_ablation(per_condition_rows):
    """Args: per_condition_rows.
    Returns: dict mapping em_iters to (summary, cluster_summary)."""
    summary_metrics = ("runtime_sec", "mse", "precision", "recall", "f1", "mcc")
    cluster_metrics = ("weight_cosine_sim_expert1_grass", "weight_cosine_sim_expert2_rock", "cluster_ari")

    condition_summaries = {}
    for em_iters, rows in per_condition_rows.items():
        moci_summary = {
            m: (float(np.mean([r[m] for r in rows])), float(np.std([r[m] for r in rows])))
            for m in summary_metrics
        }
        cluster_summary = {
            m: (float(np.mean([r[m] for r in rows])), float(np.std([r[m] for r in rows])))
            for m in cluster_metrics
        }
        condition_summaries[em_iters] = ({"MOCI": moci_summary}, cluster_summary)
    return condition_summaries


def print_em_iters_ablation(condition_summaries, n_seeds):
    """Args: condition_summaries, n_seeds.
    Returns: None."""
    print(
        f"\n{'=' * 90}\n"
        f"MOCI: effect of EM iterations on constraint detection vs. reward/cluster recovery\n"
        f"(mean over {n_seeds} seeds)\n{'=' * 90}"
    )
    print(f"{'EM iters':<10}{'MSE':>9}{'Precision':>11}{'Recall':>9}{'F1':>9}{'CosW1':>9}{'CosW2':>9}{'ClustARI':>10}")
    for em_iters in sorted(condition_summaries.keys()):
        summary, cluster_summary = condition_summaries[em_iters]
        m, c = summary["MOCI"], cluster_summary
        print(
            f"{em_iters:<10}{m['mse'][0]:>9.4f}{m['precision'][0]:>11.4f}{m['recall'][0]:>9.4f}{m['f1'][0]:>9.4f}"
            f"{c['weight_cosine_sim_expert1_grass'][0]:>9.4f}{c['weight_cosine_sim_expert2_rock'][0]:>9.4f}"
            f"{c['cluster_ari'][0]:>10.4f}"
        )


def save_em_iters_ablation(per_condition_rows, condition_summaries, out_dir: Path):
    """Args: per_condition_rows, condition_summaries, out_dir.
    Returns: None."""
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "per_seed_results.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "em_iters", "seed", "runtime_sec", "mse", "precision", "recall", "f1", "mcc",
            "weight_cosine_sim_expert1_grass", "weight_cosine_sim_expert2_rock", "cluster_ari",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for em_iters, rows in per_condition_rows.items():
            for r in rows:
                writer.writerow({"em_iters": em_iters, **r})

    gwc.save_em_iters_comparison(condition_summaries, out_dir)

    n_seeds = len(next(iter(per_condition_rows.values())))
    plot_em_iters_full_ablation(condition_summaries, n_seeds, out_dir / "em_iters_full_ablation.png")


def plot_em_iters_full_ablation(condition_summaries, n_seeds, out_path: Path):
    """Args: condition_summaries, n_seeds, out_path.
    Returns: None."""
    plt = gwc.plt
    em_list = sorted(condition_summaries.keys())

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)
    fig.patch.set_facecolor(gwc._CHART_SURFACE)

    def style_axis(ax):
        ax.set_facecolor(gwc._CHART_SURFACE)
        ax.set_xticks(em_list)
        ax.set_xlabel("EM iterations", fontsize=11, color=gwc._CHART_MUTED)
        ax.tick_params(labelsize=9.5, colors=gwc._CHART_MUTED)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color(gwc._CHART_MUTED)
        ax.spines["bottom"].set_color(gwc._CHART_MUTED)
        ax.grid(axis="y", color=gwc._CHART_GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

    def plot_series(ax, label, color, marker, get_mean_std):
        means, stds = zip(*(get_mean_std(e) for e in em_list))
        means, stds = np.array(means), np.array(stds)
        if np.any(stds > 1e-9):
            ax.fill_between(em_list, means - stds, means + stds, color=color, alpha=0.15, zorder=2)
        ax.plot(em_list, means, marker=marker, markersize=6.5, linewidth=2, color=color, label=label, zorder=3)

    style_axis(axA)
    detection_series = [
        ("MSE", "mse", "#4a3aa7", "o"),
        ("Precision", "precision", "#eda100", "s"),
        ("Recall", "recall", "#e87ba4", "^"),
        ("F1", "f1", "#008300", "D"),
    ]
    for label, key, color, marker in detection_series:
        plot_series(axA, label, color, marker, lambda e, k=key: condition_summaries[e][0]["MOCI"][k])
    axA.set_ylim(-0.03, 1.08)
    axA.set_ylabel("Score", fontsize=11, color=gwc._CHART_INK)
    axA.set_title("Constraint detection\n(σ = 0 across seeds at every setting)", fontsize=12.5, color=gwc._CHART_INK, pad=10)
    axA.legend(loc="center right", frameon=False, fontsize=9.5, labelcolor=gwc._CHART_INK)

    style_axis(axB)
    recovery_series = [
        ("Cosine sim. to W1 (Grass)", "weight_cosine_sim_expert1_grass", gwc._COLOR_GRASS, "o"),
        ("Cosine sim. to W2 (Rock)", "weight_cosine_sim_expert2_rock", gwc._COLOR_ROCK, "s"),
        ("Cluster ARI", "cluster_ari", gwc._COLOR_ARI, "^"),
    ]
    for label, key, color, marker in recovery_series:
        plot_series(axB, label, color, marker, lambda e, k=key: condition_summaries[e][1][k])
    axB.set_ylim(-0.05, 1.05)
    axB.set_ylabel("Score", fontsize=11, color=gwc._CHART_INK)
    axB.set_title("Reward / cluster recovery", fontsize=12.5, color=gwc._CHART_INK, pad=10)
    axB.legend(loc="lower right", frameon=False, fontsize=9.5, labelcolor=gwc._CHART_INK)

    fig.suptitle(
        "MOCI: constraint detection vs. reward/cluster recovery across EM iterations",
        fontsize=14.5, color=gwc._CHART_INK, x=0.02, y=0.99, ha="left",
    )
    fig.text(
        0.02, 0.90, f"mean ± std (shaded) over {n_seeds} seeds",
        fontsize=9.5, color=gwc._CHART_SECONDARY_INK, ha="left", va="top",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(out_path, dpi=300, facecolor=gwc._CHART_SURFACE)
    plt.close(fig)


SAND_LOVER_W = np.array([1.0, 0.0, 0.0, 0.0])
K_ABLATION_K_VALUES = (1, 2, 3)
K_ABLATION_N_SEEDS = 50
K_ABLATION_N_DEMOS_PER_EXPERT = 50
K_ABLATION_D_DKL = 0.05
K_ABLATION_EM_ITERS = 10
K_ABLATION_RESULTS_DIR = ROOT / "Results" / "results_additional_exp" / "k_ablation_3experts"

K_ABLATION_METRICS = ("avg_log_likelihood", "cmse", "precision", "recall", "f1", "fpr")
K_ABLATION_METRIC_LABELS = {
    "avg_log_likelihood": "Avg. marginal log-likelihood (per traj.)",
    "cmse": "CMSE (constraints)",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1 (forbidden states)",
    "fpr": "FPR",
}


def generate_pooled_demos_n_experts(mdp, water, expert_weights, n_demos_per_expert, seed):
    """Args: mdp, water, expert_weights, n_demos_per_expert, seed.
    Returns: list of pooled, shuffled demonstrations."""
    rng = np.random.default_rng(seed)
    demos = []
    for w in expert_weights:
        z = moci.backward_pass(mdp, w, water)
        demos.extend(moci.sample_traj(mdp, w, z) for _ in range(n_demos_per_expert))
    order = rng.permutation(len(demos))
    return [demos[i] for i in order]


def run_k_ablation_trial(seed, k_values, expert_weights, n_demos_per_expert, d_dkl, em_iters):
    """Args: seed, k_values, expert_weights, n_demos_per_expert, d_dkl, em_iters.
    Returns: dict mapping K to per-seed metrics."""
    water = gwc.WATER
    n_states = gwc.mdp.num_states
    with gwc._quiet():
        np.random.seed(seed)
        demos = generate_pooled_demos_n_experts(gwc.mdp, water, expert_weights, n_demos_per_expert, seed)

        rows = {}
        for K in k_values:
            inferred_c, weights, priors = moci.run_em_moci(gwc.mdp, demos, K=K, d_DKL=d_dkl, max_em_iters=em_iters)
            avg_log_likelihood = moci.calculate_joint_log_likelihood(gwc.mdp, demos, inferred_c, weights, np.array(priors))
            cmse = MLCI_exs.calculate_cmse(water, inferred_c, n_states)
            tp, fp, fn, tn, precision, recall = gwc._confusion_counts(water, inferred_c, n_states)
            f1, _ = gwc.calculate_f1_mcc(tp, fp, fn, tn)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            rows[K] = {
                "seed": seed, "K": K, "avg_log_likelihood": float(avg_log_likelihood),
                "cmse": float(cmse), "precision": float(precision), "recall": float(recall),
                "f1": float(f1), "fpr": float(fpr),
            }
    return rows


def run_k_ablation(
    seeds=None, k_values=K_ABLATION_K_VALUES, expert_weights=None,
    n_demos_per_expert=K_ABLATION_N_DEMOS_PER_EXPERT, d_dkl=K_ABLATION_D_DKL, em_iters=K_ABLATION_EM_ITERS,
):
    """Args: seeds, k_values, expert_weights, n_demos_per_expert, d_dkl, em_iters.
    Returns: dict mapping K to list of per-seed rows."""
    seeds = seeds if seeds is not None else range(K_ABLATION_N_SEEDS)
    expert_weights = expert_weights if expert_weights is not None else [gwc.W1, gwc.W2, SAND_LOVER_W]
    per_k_rows = {K: [] for K in k_values}
    for seed in seeds:
        rows = run_k_ablation_trial(seed, k_values, expert_weights, n_demos_per_expert, d_dkl, em_iters)
        for K, row in rows.items():
            per_k_rows[K].append(row)
    return per_k_rows


def summarize_k_ablation(per_k_rows):
    """Args: per_k_rows.
    Returns: dict mapping K to metric (mean, std) pairs."""
    return {
        K: {
            m: (float(np.mean([r[m] for r in rows])), float(np.std([r[m] for r in rows])))
            for m in K_ABLATION_METRICS
        }
        for K, rows in per_k_rows.items()
    }


def print_k_ablation(summary, n_seeds):
    """Args: summary, n_seeds.
    Returns: None."""
    k_values = sorted(summary.keys())
    print(
        f"\n{'=' * 90}\n"
        f"Ablation: latent mixture vs. single preference model "
        f"(3 pooled experts: Grass, Rock, Sand; Water = hard constraint)\n"
        f"Means ± std over {n_seeds} seeds\n{'=' * 90}"
    )
    header = f"{'Metric':<42}" + "".join(f"K={K}".center(18) for K in k_values)
    print(header)
    print("-" * len(header))
    for m in K_ABLATION_METRICS:
        cells = "".join(f"{summary[K][m][0]:.4f} ± {summary[K][m][1]:.4f}".center(18) for K in k_values)
        print(f"{K_ABLATION_METRIC_LABELS[m]:<42}{cells}")


def save_k_ablation(per_k_rows, summary, out_dir: Path, n_seeds):
    """Args: per_k_rows, summary, out_dir, n_seeds.
    Returns: None."""
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "per_seed_results.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["K", "seed", *K_ABLATION_METRICS]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for K, rows in per_k_rows.items():
            for r in rows:
                writer.writerow({"K": K, "seed": r["seed"], **{m: r[m] for m in K_ABLATION_METRICS}})

    with (out_dir / "summary_mean_std.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["K", "metric", "mean", "std"])
        for K, metrics in summary.items():
            for m, (mean, std) in metrics.items():
                writer.writerow([K, m, mean, std])

    k_values = sorted(summary.keys())
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        rf"  \caption{{Pooled heterogeneous demonstrations (Grass/Rock/Sand experts, Water hard constraint): "
        rf"single-preference MOCI ($K{{=}}1$) vs.\ mixture MOCI ($K{{=}}2,3$). "
        rf"Means $\pm$ std.\ over {n_seeds} seeds.}}",
        r"  \label{tab:exp1_k_ablation_3experts}",
        rf"  \begin{{tabular}}{{l{'c' * len(k_values)}}}",
        r"    \hline",
        rf"    Metric & {' & '.join(f'$K{{=}}{K}$' for K in k_values)} \\",
        r"    \hline",
    ]
    for m in K_ABLATION_METRICS:
        cells = " & ".join(rf"${summary[K][m][0]:.4f} \pm {summary[K][m][1]:.4f}$" for K in k_values)
        lines.append(rf"    {K_ABLATION_METRIC_LABELS[m]} & {cells} \\")
    lines += [r"    \hline", r"  \end{tabular}", r"\end{table}"]
    (out_dir / "k_ablation_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    print("=== Oracle-reward ablation ===")
    results, water, n_states = run_ablation()
    print_results(results, water, n_states)

    print(f"\n\n=== EM-iterations ablation (MOCI only, {len(gwc.SEEDS)} seeds per setting) ===")
    per_condition_rows = run_em_iters_ablation()
    condition_summaries = summarize_em_iters_ablation(per_condition_rows)
    print_em_iters_ablation(condition_summaries, len(gwc.SEEDS))
    save_em_iters_ablation(per_condition_rows, condition_summaries, EM_ITERS_RESULTS_DIR)
    print(f"\nSaved EM-iterations ablation results to: {EM_ITERS_RESULTS_DIR}")

    print(f"\n\n=== K ablation, 3 pooled experts (Grass/Rock/Sand), {K_ABLATION_N_SEEDS} seeds per K ===")
    per_k_rows = run_k_ablation()
    k_summary = summarize_k_ablation(per_k_rows)
    print_k_ablation(k_summary, K_ABLATION_N_SEEDS)
    save_k_ablation(per_k_rows, k_summary, K_ABLATION_RESULTS_DIR, K_ABLATION_N_SEEDS)
    print(f"\nSaved K-ablation (3 experts) results to: {K_ABLATION_RESULTS_DIR}")
