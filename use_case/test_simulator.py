import gymnasium
from gymnasium.envs.registration import register
from mo_reward import mo_risk_diff_reward
from simglucose.controller.basal_bolus_exercise_ctrller import BBExerciseController, DiscretizedBBExerciseController
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.controller.mo_basal_bolus_ctrller import MOBBExerciseController, DiscretizedMOBBExerciseController

patient_name = "adolescent#002"

seed = 0

register(
    id="simglucose/adolescent2-v0",
    entry_point="simglucose.envs:DiscBasalBolusT1DSimGymnasiumEnv",
    max_episode_steps=1440,
    kwargs={
        "patient_name": patient_name,
        "reward_fun": mo_risk_diff_reward,
        "seed": seed
        },
)

env = gymnasium.make("simglucose/adolescent2-v0", render_mode="human")

controller = DiscretizedMOBBExerciseController(env=env, preference='bolus-adverse')
observation, info = env.reset()
reward, terminated = 0, False
for t in range(1000):
    env.render()
    action = controller.policy(observation, reward, terminated, patient_name=patient_name, meal=info.get('meal'), sample_time=info.get('sample_time'))
    observation, reward, terminated, truncated, info = env.step(action)
    print(
        f"Step {t}: action {action}, observation {observation}, reward {reward}, terminated {terminated}, truncated {truncated}, info {info}"
    )
    if terminated or truncated:
        print("Episode finished after {} timesteps".format(t + 1))
        break
    