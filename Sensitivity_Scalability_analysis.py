import os
import time
import numpy as np
import matplotlib.pyplot as plt
import MOCI_IRL as moci
import gridworld as gw

# ==========================================
# 1. CORE ALGORITHM FUNCTIONS
# ==========================================

def calculate_fpr(ground_truth_c, inferred_c, total_states):
    """Calculates the False Positive Rate (FPR)."""
    fp, tn = 0, 0
    for state in range(total_states):
        is_true = state in ground_truth_c
        is_inferred = state in inferred_c
        
        if not is_true and is_inferred:
            fp += 1
        elif not is_true and not is_inferred:
            tn += 1
            
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0

# ==========================================
# 2. PLOTTING AND SAVING FUNCTIONS
# ==========================================

def plot_fpr_vs_demos(demo_sizes, fpr_results, thresholds, save_path):
    """Plots FPR vs Dataset Size over different thresholds."""
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
    """Plots FPR against grid size using strictly fixed constraints."""
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
    """Plots execution time vs grid size comparing two trajectory lengths."""
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

# ==========================================
# 3. EXECUTION SCRIPT
# ==========================================

def run_sensitivity_and_scalability_experiments():
    results_dir = "Results"
    os.makedirs(results_dir, exist_ok=True)
    
    w1 = np.array([1.0, 3.0, -1.0, -10.0])
    w2 = np.array([1.0, -1.0, 3.0, -10.0])
    thresholds_to_test = [0.01, 0.05, 0.5, 2.0]
    grid_sizes_to_test = list(range(5, 11))
    max_demos=50  # number of demonstrations to test in Experiment 1
    step=4 #step size for increasing demonstrations in Experiment 1
    
    # ---------------------------------------------------------
    # EXPERIMENT 1: Effect of Dataset Size on FPR (Fixed Grid)
    # ---------------------------------------------------------
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
        # Generating fresh trajectories as dataset size increases
        D_current = [moci.sample_traj(mdp_exp1, w1, z1_exp1) for _ in range(half_n)] + \
                    [moci.sample_traj(mdp_exp1, w2, z2_exp1) for _ in range(n_demos - half_n)]
        
        for t in thresholds_to_test:
            inferred_c, _, _ = moci.run_em_moci(mdp_exp1, D_current, K=2, d_DKL=t, max_em_iters=3)
            fpr = calculate_fpr(water_exp1, inferred_c, mdp_exp1.num_states)
            fpr_results_demos[t].append(fpr)
            
    plot_fpr_vs_demos(demo_sizes, fpr_results_demos, thresholds_to_test, save_path="Results/FPR_vs_DatasetSize.png")

    # ---------------------------------------------------------
    # EXPERIMENT 2: Robustness - FPR vs Grid Size (Fixed Constraints)
    # ---------------------------------------------------------
    print("\n--- Running Experiment 2: Robustness across Grids ---")
    # Using fixed indices that are valid even on the smallest (5x5) grid
    fixed_water = [12, 13,14, 18]
    fixed_grass =  [4, 8,9,11,17]
    fixed_rocks = [10, 12, 16, 22]
    
    FIXED_DEMOS = 20
    fpr_across_grids = {t: [] for t in thresholds_to_test}
    
    for size in grid_sizes_to_test:
        print(f"  Evaluating fixed constraints on {size}x{size} grid...")
        mdp_exp2 = gw.CustomizableFeatureMDP(size, fixed_water, fixed_grass, fixed_rocks)
        
        z1_exp2 = moci.backward_pass(mdp_exp2, w1, fixed_water)
        z2_exp2 = moci.backward_pass(mdp_exp2, w2, fixed_water)
        
        D_fixed = [moci.sample_traj(mdp_exp2, w1, z1_exp2) for _ in range(FIXED_DEMOS // 2)] + \
                  [moci.sample_traj(mdp_exp2, w2, z2_exp2) for _ in range(FIXED_DEMOS // 2)]
                  
        for t in thresholds_to_test:
            inferred_c, _, _ = moci.run_em_moci(mdp_exp2, D_fixed, K=2, d_DKL=t, max_em_iters=3)
            fpr = calculate_fpr(fixed_water, inferred_c, mdp_exp2.num_states)
            fpr_across_grids[t].append(fpr)
            
    plot_fpr_vs_gridsize(grid_sizes_to_test, fpr_across_grids, thresholds_to_test, save_path="Results/FPR_vs_GridSize.png")

# ---------------------------------------------------------
    # EXPERIMENT 3: Scalability - Runtime vs Grid Size (Dynamic Horizons)
    # ---------------------------------------------------------
    print("\n--- Running Experiment 3: Runtime over Dynamic Trajectory Lengths ---")
    def plot_runtime_vs_gridsize_dynamic(grid_sizes, runtimes_short, runtimes_long, save_path):
        """Plots execution time vs grid size comparing two dynamic trajectory lengths."""
        plt.figure(figsize=(8, 5))
        
        plt.plot(grid_sizes, runtimes_short, marker='o', color='#3498DB', linewidth=2, 
                markersize=8, label='Short Trajectory (~2N steps)')
        plt.plot(grid_sizes, runtimes_long, marker='s', color='#E74C3C', linewidth=2, 
                markersize=8, label='Long Trajectory (~5N steps)')
        
        plt.xlabel('Grid Size (N x N)', fontsize=12)
        plt.ylabel('Execution Time (seconds)', fontsize=12)
        plt.title('Scalability: Run-time vs. Grid Size (Dynamic Horizons)', fontsize=14)
        plt.xticks(grid_sizes, [f"{s}x{s}" for s in grid_sizes])
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    runtimes_short = []
    runtimes_long = []
    
    for size in grid_sizes_to_test:
        print(f"  Measuring runtime on {size}x{size} grid...")
        mdp_exp3 = gw.CustomizableFeatureMDP(size, fixed_water, fixed_grass, fixed_rocks)
        
        # 1. Define dynamic horizons based on current grid size (N)
        # Shortest path corner-to-corner is 2N - 2. We use 2N.
        H_short = 2 * size  
        # Long path wanders the grid. We use 5N.
        H_long = 5 * size   
        
        # 2. Test both horizons
        for horizon, runtime_list in [(H_short, runtimes_short), (H_long, runtimes_long)]:
            # Set the dynamic trajectory length in your MDP
            mdp_exp3.horizon = horizon  
            
            # Recalculate partition functions for the new horizon
            z1_exp3 = moci.backward_pass(mdp_exp3, w1, fixed_water)
            z2_exp3 = moci.backward_pass(mdp_exp3, w2, fixed_water)
            
            # Generate demonstrations using the specific trajectory length
            D_timing = [moci.sample_traj(mdp_exp3, w1, z1_exp3) for _ in range(10)] + \
                       [moci.sample_traj(mdp_exp3, w2, z2_exp3) for _ in range(10)]
            
            # Measure execution time
            start_time = time.time()
            moci.run_em_moci(mdp_exp3, D_timing, K=2, d_DKL=0.05, max_em_iters=2) # Reduced iters for timing
            exec_time = time.time() - start_time
            
            # Save to the corresponding list
            runtime_list.append(exec_time)

    # 3. Plot the results using the updated dynamic plotting function
    plot_runtime_vs_gridsize_dynamic(
        grid_sizes_to_test, 
        runtimes_short, 
        runtimes_long, 
        os.path.join(results_dir, "Runtime_vs_GridSize_Dynamic_Horizons.png")
    )
    print("\nAll experiments finished successfully! Check the Results folder.")