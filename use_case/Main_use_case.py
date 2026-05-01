
import numpy as np

import gymnasium
from gymnasium.envs.registration import register
from simglucose.controller.basal_bolus_exercise_ctrller import BBExerciseController, DiscretizedBBExerciseController
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.controller.mo_basal_bolus_ctrller import MOBBExerciseController, DiscretizedMOBBExerciseController
from mo_reward import mo_risk_diff_reward
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import MOCI_IRL as moci

# --- STEP 3: DEFINE DEMONSTRATION COUNTS ---
# N_DEMOS_EXPERT1 = 10
# N_DEMOS_EXPERT2 = 10
N_DEMOS_EXPERT1 = 2
N_DEMOS_EXPERT2 = 2

# N_STEPS = 1440 # 24 hours at 1 min interval
N_STEPS = 20

patient_name = "adolescent#002"
seed = 0

register(
    id="simglucose/adolescent2-v0",
    entry_point="simglucose.envs:DiscBasalBolusT1DSimGymnasiumEnv",
    max_episode_steps=N_STEPS,
    kwargs={
        "patient_name": patient_name,
        "reward_fun": mo_risk_diff_reward,
        "seed": seed
        },
)

env = gymnasium.make("simglucose/adolescent2-v0", render_mode="human")
env.horizon = N_STEPS  # Ensure the environment's horizon matches our desired number of steps for demos


controller = DiscretizedMOBBExerciseController(env=env, preference='bolus-adverse')

def get_demo(controller):
    """
    Function to obtain a single expert trajectory demonstration for the use case custom simglucose environment.
    """
    # TODO: how to reset env to reasonable new state? E.g. reset random seed or adjust timing?
    env.reset()
    env.seed = 0
    demo = []
    observation, info = env.reset()
    reward, terminated = 0, False
    for t in range(N_STEPS):
        action = controller.policy(observation, reward, terminated, patient_name=patient_name, meal=info.get('meal'), sample_time=info.get('sample_time'))
        next_observation, reward, terminated, truncated, info = env.step(action)
        demo.append((observation, action))
        observation = next_observation
        if terminated or truncated:
            print("Demo finished after {} timesteps".format(t + 1))
            break
    return demo

def get_demos(controller, n_demos):
    """
    Function to obtain expert trajectory demonstrations for the use case custom simglucose environment.
    """
    env.reset()
    env.seed = 0
    demos = []
    for _ in range(n_demos):
        demos.append(get_demo(controller))
    return demos
    

# ==========================================
# EXECUTION SCRIPT
# ==========================================
if __name__ == "__main__":
    # Example: mdp = CustomizableFeatureMDP(GRID_SIZE, WATER, GRASS, ROCKS)
    # Example: all_demos = [...]
    controller = DiscretizedMOBBExerciseController(env=env, preference='hyper-adverse')
    expert_1_demos = get_demos(controller, N_DEMOS_EXPERT1)
    expert_1_demos = get_demos(controller, N_DEMOS_EXPERT1)
    controller = DiscretizedMOBBExerciseController(env=env, preference='bolus-adverse')
    expert_2_demos = get_demos(controller, N_DEMOS_EXPERT2)
    
    all_demos = expert_1_demos + expert_2_demos
    
    inferred_c, final_weights, final_priors = moci.run_em_moci(env, all_demos, K=2, d_DKL=0.05, max_em_iters=10)

    print(f"Algorithm Inferred Constraints: {list(inferred_c)}")

    # We use the same 'resp' array to keep the trajectory colors consistent.
    # Passing 'inferred_c' will trigger the red hatched boxes in your plotting function.
    title_inferred = "MOCI Inferred Constraints (Red Hatched)"



    gw.plot_grid_setup( mdp=mdp,  title=title_inferred, demos=all_demos, resp=resp, inf_c=inferred_c)  # <--- This replaces the ground-truth visualization with the algorithm's output


    print("Inferred Constraints:", sorted(list(inferred_c)))
    print("Final Weights:", final_weights)
    print("Final Priors:", final_priors)

    # Assuming final_weights[0] mapped to the Grass-Lover cluster
    print("Learned Preferences for Cluster 1:", np.round(final_weights[0], 2))
    # Expected output: Something like [ 0.1,  2.5, -1.8, -0.5]
    # High positive weight for index 1 (Grass), negative for index 2 (Rock)

    # Assuming final_weights[1] mapped to the Rock-Lover cluster
    print("Learned Preferences for Cluster 2:", np.round(final_weights[1], 2))
    # Expected output: Something like [-0.2, -2.1,  3.0, -0.4]
    # High positive weight for index 2 (Rock), negative for index 1 (Grass)

    gw.plot_preference_recovery(w1, w2, final_weights, features=['Sand', 'Grass', 'Rocks', 'Water'])

    SSA.run_sensitivity_and_scalability_experiments()



