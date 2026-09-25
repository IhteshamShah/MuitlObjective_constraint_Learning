"""Sensitivity and scalability analyses for MOCI on gridworld environments.
Computes the false positive rate (FPR) between ground-truth and inferred
hazard constraints, and runs three experiments: FPR vs. number of expert
demonstrations at a fixed grid size, FPR vs. grid size with fixed
constraints, and MOCI runtime vs. grid size across short/long trajectory
horizons. Each experiment saves a plot (and, for the runtime experiment, a
CSV) to a results directory. When run as a standalone script, results are
written under Results/results_additional_exp.
"""

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import MOCI_IRL as moci
import use_case.GridWorld.gridworld_Env as gw

def calculate_fpr(ground_truth_c, inferred_c, total_states):
    """
    Parameters: ground_truth_c, inferred_c, total_states.
    Returns: false positive rate as a float.
    """
    fp, tn = 0, 0
    for state in range(total_states):
        is_true = state in ground_truth_c
        is_inferred = state in inferred_c

        if not is_true and is_inferred:
            fp += 1
        elif not is_true and not is_inferred:
            tn += 1

    return fp / (fp + tn) if (fp + tn) > 0 else 0.0

def plot_fpr_vs_demos(demo_sizes, fpr_results, thresholds, save_path):
    """
    Parameters: demo_sizes, fpr_results, thresholds, save_path.
    Returns: None; saves the plot to save_path.
    """
    plt.figure(figsize=(8, 5))
    markers = ['o', 's', '^', 'D']
    colors = ['#E74C3C', '#F39C12', '#2ECC71', '#3498DB']

    for i, t in enumerate(thresholds):
        plt.plot(demo_sizes, fpr_results[t], marker=markers[i], color=colors[i],
                 linewidth=2, markersize=8, label=f'd_DKL = {t}')

    plt.xlabel('Number of Expert Demonstrations (|D|)', fontsize=12)
    plt.ylabel('False Positive Rate (FPR)', fontsize=12)
    plt.title('Effect of Dataset Size on FPR', fontsize=14)
    plt.ylim([-0.05, 1.05])
    plt.xticks(demo_sizes)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Divergence Threshold', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_fpr_vs_gridsize(grid_sizes, fpr_dict, thresholds, save_path):
    """
    Parameters: grid_sizes, fpr_dict, thresholds, save_path.
    Returns: None; saves the plot to save_path.
    """
    plt.figure(figsize=(8, 5))
    markers = ['o', 's', '^', 'D']
    colors = ['#E74C3C', '#F39C12', '#2ECC71', '#3498DB']

    for i, t in enumerate(thresholds):
        plt.plot(grid_sizes, fpr_dict[t], marker=markers[i], color=colors[i],
                 linewidth=2, markersize=8, label=f'd_DKL = {t}')

    plt.xlabel('Grid Size (N x N)', fontsize=12)
    plt.ylabel('False Positive Rate (FPR)', fontsize=12)
    plt.title('Robustness: FPR vs. Grid Size (Fixed Constraints)', fontsize=14)
    plt.ylim([-0.05, 1.05])
    plt.xticks(grid_sizes, [f"{s}x{s}" for s in grid_sizes])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Divergence Threshold', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_runtime_vs_gridsize_horizons(grid_sizes, runtimes_short, runtimes_long, H_short, H_long, save_path):
    """
    Parameters: grid_sizes, runtimes_short, runtimes_long, H_short, H_long, save_path.
    Returns: None; saves the plot to save_path.
    """
    plt.figure(figsize=(8, 5))

    plt.plot(grid_sizes, runtimes_short, marker='o', color='#3498DB', linewidth=2,
             markersize=8, label=f'Short Trajectory (H={H_short})')
    plt.plot(grid_sizes, runtimes_long, marker='s', color='#E74C3C', linewidth=2,
             markersize=8, label=f'Long Trajectory (H={H_long})')

    plt.xlabel('Grid Size (N x N)', fontsize=12)
    plt.ylabel('Execution Time (seconds)', fontsize=12)
    plt.title('Scalability: Run-time vs. Grid Size over Trajectory Lengths', fontsize=14)
    plt.xticks(grid_sizes, [f"{s}x{s}" for s in grid_sizes])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def run_sensitivity_and_scalability_experiments(
    results_dir="Results",
    thresholds_to_test=None,
    grid_sizes_to_test=None,
    max_demos=50,
    step=4,
    fixed_demos=20,
    timing_em_iters=2,
    em_iters_exp12=3,
):
    """
    Parameters: results_dir, thresholds_to_test, grid_sizes_to_test, max_demos,
    step, fixed_demos, timing_em_iters, em_iters_exp12.
    Returns: None; saves plots and CSVs to results_dir.
    """
    os.makedirs(results_dir, exist_ok=True)

    w1 = np.array([1.0, 3.0, -1.0, -10.0])
    w2 = np.array([1.0, -1.0, 3.0, -10.0])
    thresholds_to_test = thresholds_to_test if thresholds_to_test is not None else [0.01, 0.05, 0.5, 2.0]
    grid_sizes_to_test = grid_sizes_to_test if grid_sizes_to_test is not None else list(range(5, 11))

    print("\n--- Running Experiment 1: FPR vs Dataset Size ---")
    fixed_grid_size = 6
    water_exp1 = [14, 15,18,19,27]
    grass=[4, 8,9,11,17]
    rocks=[10, 12, 16, 22, 28]
    mdp_exp1 = gw.CustomizableFeatureMDP(fixed_grid_size, water_exp1, grass, rocks)

    demo_sizes = [1] + list(range(step, max_demos + 1, step))
    fpr_results_demos = {t: [] for t in thresholds_to_test}

    z1_exp1 = moci.backward_pass(mdp_exp1, w1, water_exp1)
    z2_exp1 = moci.backward_pass(mdp_exp1, w2, water_exp1)

    for n_demos in demo_sizes:
        print(f"  Generating {n_demos} new trajectories...")
        half_n = n_demos // 2
        D_current = [moci.sample_traj(mdp_exp1, w1, z1_exp1) for _ in range(half_n)] + \
                    [moci.sample_traj(mdp_exp1, w2, z2_exp1) for _ in range(n_demos - half_n)]

        for t in thresholds_to_test:
            inferred_c, _, _ = moci.run_em_moci(mdp_exp1, D_current, K=2, d_DKL=t, max_em_iters=em_iters_exp12)
            fpr = calculate_fpr(water_exp1, inferred_c, mdp_exp1.num_states)
            fpr_results_demos[t].append(fpr)

    plot_fpr_vs_demos(
        demo_sizes,
        fpr_results_demos,
        thresholds_to_test,
        save_path=os.path.join(results_dir, "FPR_vs_DatasetSize.png"),
    )

    print("\n--- Running Experiment 2: Robustness across Grids ---")
    fixed_water = [12, 13,14, 18]
    fixed_grass =  [4, 8,9,11,17]
    fixed_rocks = [10, 12, 16, 22]

    FIXED_DEMOS = fixed_demos
    fpr_across_grids = {t: [] for t in thresholds_to_test}

    for size in grid_sizes_to_test:
        print(f"  Evaluating fixed constraints on {size}x{size} grid...")
        mdp_exp2 = gw.CustomizableFeatureMDP(size, fixed_water, fixed_grass, fixed_rocks)

        z1_exp2 = moci.backward_pass(mdp_exp2, w1, fixed_water)
        z2_exp2 = moci.backward_pass(mdp_exp2, w2, fixed_water)

        D_fixed = [moci.sample_traj(mdp_exp2, w1, z1_exp2) for _ in range(FIXED_DEMOS // 2)] + \
                  [moci.sample_traj(mdp_exp2, w2, z2_exp2) for _ in range(FIXED_DEMOS // 2)]

        for t in thresholds_to_test:
            inferred_c, _, _ = moci.run_em_moci(mdp_exp2, D_fixed, K=2, d_DKL=t, max_em_iters=em_iters_exp12)
            fpr = calculate_fpr(fixed_water, inferred_c, mdp_exp2.num_states)
            fpr_across_grids[t].append(fpr)

    plot_fpr_vs_gridsize(
        grid_sizes_to_test,
        fpr_across_grids,
        thresholds_to_test,
        save_path=os.path.join(results_dir, "FPR_vs_GridSize.png"),
    )

    print("\n--- Running Experiment 3: Runtime over Dynamic Trajectory Lengths ---")
    def plot_runtime_vs_gridsize_dynamic(grid_sizes, runtimes_short, runtimes_long, save_path):
        """
        Parameters: grid_sizes, runtimes_short, runtimes_long, save_path.
        Returns: None; saves the plot to save_path.
        """
        SURFACE = "#fcfcfb"
        GRID_COLOR = "#e1e0d9"
        MUTED = "#898781"
        INK = "#0b0b0b"
        COLOR_SHORT = "#2a78d6"
        COLOR_LONG = "#eb6834"

        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

        fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
        fig.patch.set_facecolor(SURFACE)
        ax.set_facecolor(SURFACE)

        ax.plot(
            grid_sizes, runtimes_short, marker="o", markersize=7.5, linewidth=2.2,
            color=COLOR_SHORT, label="Short trajectory (~2N steps)", zorder=3,
        )
        ax.plot(
            grid_sizes, runtimes_long, marker="s", markersize=7.5, linewidth=2.2,
            color=COLOR_LONG, label="Long trajectory (~5N steps)", zorder=3,
        )

        ax.set_xlabel("Grid size (N × N)", fontsize=12, color=INK)
        ax.set_ylabel("Execution time (seconds)", fontsize=12, color=INK)
        ax.set_title("Scalability: MOCI run-time vs. grid size", fontsize=14.5, color=INK, pad=14, loc="left")
        ax.set_xticks(grid_sizes)
        ax.set_xticklabels([f"{s}×{s}" for s in grid_sizes], fontsize=10.5, color=MUTED)
        ax.tick_params(axis="y", labelsize=10.5, colors=MUTED)
        ax.tick_params(axis="x", length=0)

        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color(MUTED)
        ax.spines["bottom"].set_color(MUTED)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=1.0, zorder=0)
        ax.set_axisbelow(True)

        ax.legend(loc="upper left", frameon=False, fontsize=10.5, labelcolor=INK)

        fig.tight_layout()
        fig.savefig(save_path, dpi=300, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)

    runtimes_short = []
    runtimes_long = []

    for size in grid_sizes_to_test:
        print(f"  Measuring runtime on {size}x{size} grid...")
        mdp_exp3 = gw.CustomizableFeatureMDP(size, fixed_water, fixed_grass, fixed_rocks)

        H_short = 2 * size
        H_long = 5 * size

        for horizon, runtime_list in [(H_short, runtimes_short), (H_long, runtimes_long)]:
            mdp_exp3.horizon = horizon

            z1_exp3 = moci.backward_pass(mdp_exp3, w1, fixed_water)
            z2_exp3 = moci.backward_pass(mdp_exp3, w2, fixed_water)

            D_timing = [moci.sample_traj(mdp_exp3, w1, z1_exp3) for _ in range(10)] + \
                       [moci.sample_traj(mdp_exp3, w2, z2_exp3) for _ in range(10)]

            start_time = time.time()
            moci.run_em_moci(mdp_exp3, D_timing, K=2, d_DKL=0.05, max_em_iters=timing_em_iters)
            exec_time = time.time() - start_time

            runtime_list.append(exec_time)

    plot_runtime_vs_gridsize_dynamic(
        grid_sizes_to_test,
        runtimes_short,
        runtimes_long,
        os.path.join(results_dir, "Runtime_vs_GridSize_Dynamic_Horizons.png")
    )

    csv_path = os.path.join(results_dir, "Runtime_vs_GridSize_Dynamic_Horizons.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["grid_size", "horizon_condition", "horizon_steps", "runtime_sec"])
        for size, rt_short, rt_long in zip(grid_sizes_to_test, runtimes_short, runtimes_long):
            writer.writerow([size, "short (2N)", 2 * size, rt_short])
            writer.writerow([size, "long (5N)", 5 * size, rt_long])
    print(f"Saved runtime-vs-gridsize data to: {csv_path}")

    print("\nAll experiments finished successfully! Check the Results folder.")


if __name__ == "__main__":
    RESULTS_DIR = ROOT / "Results" / "results_additional_exp"
    run_sensitivity_and_scalability_experiments(results_dir=str(RESULTS_DIR))
