This is the T1 diabetes use case for MOCI.

The original UvA/PADOVA simglucose simulator has been extended in the following way:
1. Increased blood glucose uptake during physical exercise based on %VO2Max
2. Extension of action space to include basal, bolus, and exercise intensity (measured as %VO2Max)
3. Discretization of states and actions for both environment and rule-based controller (adapted from BBController), including temporal discretization error modulation
4. inclusion of (action-based) constraints on the environment and adherence to constraints in rule-based controller. By default, the following constraints are applied:
    1. exercise intensity must be ``<= .9`` %VO2Max
    2. bolus dosage must be ``<=20``
    3. basal dosage must be ``<=5``
5. Extension of reward function to multi-objective setting. this reward consists of four components: ``[w_risk_hyper, w_risk_hypo, w_bolus, w_exercise]``
    1. the risk-components are state-based and express whether the patient is at risk of either a hyper or hypo
    2. the final components indicate a preference for actions: whether the prefers a bolus dosage or exercise.
    Note: the weights are not explicitly implemented since the controller is a rule-based controller aimed to mimick real patient behavior.
6. Inclusion of persona's in the rule-based controller:
    1. a ``hyper-adverse`` persona that aims to avoid hyperglycema by high bolus dosage and high exercise intensity
    2. a ``hypo-adverse`` persona that aims to avoid hypoglycema by low bolus dosage and low exercise intensity
    3. a ``bolus-adverse`` persona that aims to avoid bolus dosages and prefers high exercise instead
    4. an ``exercise-adverse`` persona that aims to avoids exercise and prefers higher bolus dosages


# Installation
This use case only works with Python 3.10 as far as has been tested. Later vsions may not work due to simglucose incompatibility.

The easiest way to install is to install editable files as follows in your favorite virtual environment:
```bash
cd _simglucose
pip install -e .
pip install setuptools\<=80.10.2 # necessary for pkg_resources removal
```

To test the installation:
```bash
python test_simulator.py
```

# Usage
To generate a trajectory
```python
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
episode_length - 1440 # minutes in a day
for t in range(episode_length):
    env.render()
    action = controller.policy(observation, reward, terminated, patient_name=patient_name, meal=info.get('meal'), sample_time=info.get('sample_time'))
    observation, reward, terminated, truncated, info = env.step(action)
    print(
        f"Step {t}: action {action}, observation {observation}, reward {reward}, terminated {terminated}, truncated {truncated}, info {info}"
    )
    if terminated or truncated:
        print("Episode finished after {} timesteps".format(t + 1))
        break
```

Constraints can be set on the discretized action space.
```python
env = gymnasium.make("simglucose/adolescent2-v0", render_mode="human")
# this places a hard constraint on providing a dosage of bins (0, 0, 0)
env.add_constraints([
    (1, 0, 0),
])
```
If you continue to use that environment instance, make sure you reset the constraints!
```python
env.relax_constraints(reset=False) # set to True to reset to the ground truth constraints
```