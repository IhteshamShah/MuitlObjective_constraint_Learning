from importlib import resources
import logging

from simglucose.simulation.env import T1DSimEnv as _T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.controller.base import ExerciseAction as Action
import numpy as np
import pkg_resources
import gym
from gymnasium import spaces
from gym.utils import seeding
from datetime import datetime
import gymnasium
from typing import Iterable
import functools

logger = logging.getLogger(__name__)

# Modern resource access
package_path = resources.files("simglucose")
PATIENT_PARA_FILE = str(package_path / "simglucose" / "params" / "vpatient_params.csv")

class BasalBolusT1DSimEnv(gym.Env):
    """
    A wrapper of simglucose.simulation.env.T1DSimEnv to support gym API with basal/bolus action space.
    """

    metadata = {"render.modes": ["human"]}

    SENSOR_HARDWARE = "Dexcom"
    INSULIN_PUMP_HARDWARE = "Insulet"

    def __init__(
        self, patient_name=None, custom_scenario=None, reward_fun=None, seed=None
    ):
        """
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        """
        # have to hard code the patient_name, gym has some interesting
        # error when choosing the patient
        if patient_name is None:
            patient_name = ["adolescent#001"]

        self.patient_name = patient_name
        self.reward_fun = reward_fun
        self.np_random, _ = seeding.np_random(seed=seed)
        self.custom_scenario = custom_scenario
        self.env, _, _, _ = self._create_env()

    def _step(self, action: float):
        # This gym only controls basal insulin
        act = Action(basal=action[0], bolus=action[1], exercise_intensity=action[2])
        if self.reward_fun is None:
            return self.env.step(act)
        return self.env.step(act, reward_fun=self.reward_fun)

    def _raw_reset(self):
        return self.env.reset()

    def _reset(self):
        self.env, _, _, _ = self._create_env()
        obs, _, _, _ = self.env.reset()
        return obs

    def _seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        self.env, seed2, seed3, seed4 = self._create_env()
        return [seed1, seed2, seed3, seed4]

    def _create_env(self):
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(self.np_random.randint(0, 1000)) % 2**31
        seed3 = seeding.hash_seed(seed2 + 1) % 2**31
        seed4 = seeding.hash_seed(seed3 + 1) % 2**31

        hour = self.np_random.randint(low=0.0, high=24.0)
        start_time = datetime(2018, 1, 1, hour, 0, 0)

        if isinstance(self.patient_name, list):
            patient_name = self.np_random.choice(self.patient_name)
            patient = T1DPatient.withName(patient_name, random_init_bg=True, seed=seed4)
        else:
            patient = T1DPatient.withName(
                self.patient_name, random_init_bg=True, seed=seed4
            )

        if isinstance(self.custom_scenario, list):
            scenario = self.np_random.choice(self.custom_scenario)
        else:
            scenario = (
                RandomScenario(start_time=start_time, seed=seed3)
                if self.custom_scenario is None
                else self.custom_scenario
            )

        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=seed2)
        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        env = _T1DSimEnv(patient, sensor, pump, scenario)
        return env, seed2, seed3, seed4

    def _render(self, mode="human", close=False):
        self.env.render(close=close)

    def _close(self):
        super()._close()
        self.env._close_viewer()

    @property
    def action_space(self):
        ubasal = self.env.pump._params["max_basal"]
        ubolus = self.env.pump._params["max_bolus"]
        uexercise = self.max_exercise_intensity # max in terms of %VO2 max
        return spaces.Box(low=(0, 0, 0), high=(ubasal, ubolus, uexercise), shape=(3,), dtype=np.float32)

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=1000, shape=(1,))

    @property
    def max_basal(self):
        return self.env.pump._params["max_basal"]
    
    @property   
    def max_bolus(self):
        return self.env.pump._params["max_bolus"]

    @property
    def max_exercise_intensity(self):
        return 1.0 # max in terms of %VO2 max

class DiscBasalBolusT1DSimEnv(gym.Env):
    """
    A discretized wrapper of simglucose.simulation.env.T1DSimEnv to support gym API with basal/bolus action space.
    """

    metadata = {"render.modes": ["human"]}

    SENSOR_HARDWARE = "Dexcom"
    INSULIN_PUMP_HARDWARE = "Insulet"

    def __init__(
        self, patient_name=None, custom_scenario=None, reward_fun=None, seed=None
    ):
        """
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        """
        # have to hard code the patient_name, gym has some interesting
        # error when choosing the patient
        if patient_name is None:
            patient_name = ["adolescent#001"]

        self.patient_name = patient_name
        self.reward_fun = reward_fun
        self.np_random, _ = seeding.np_random(seed=seed)
        self.custom_scenario = custom_scenario
        self.env, _, _, _ = self._create_env()

    def _step(self, action: float):
        # This gym only controls basal insulin
        act = Action(basal=action[0], bolus=action[1], exercise_intensity=action[2])
        if self.reward_fun is None:
            return self.env.step(act)
        return self.env.step(act, reward_fun=self.reward_fun)

    def _raw_reset(self):
        return self.env.reset()

    def _reset(self):
        self.env, _, _, _ = self._create_env()
        obs, _, _, _ = self.env.reset()
        return obs

    def _seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        self.env, seed2, seed3, seed4 = self._create_env()
        return [seed1, seed2, seed3, seed4]

    def _create_env(self):
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(self.np_random.randint(0, 1000)) % 2**31
        seed3 = seeding.hash_seed(seed2 + 1) % 2**31
        seed4 = seeding.hash_seed(seed3 + 1) % 2**31

        hour = self.np_random.randint(low=0.0, high=24.0)
        start_time = datetime(2018, 1, 1, hour, 0, 0)

        if isinstance(self.patient_name, list):
            patient_name = self.np_random.choice(self.patient_name)
            patient = T1DPatient.withName(patient_name, random_init_bg=True, seed=seed4)
        else:
            patient = T1DPatient.withName(
                self.patient_name, random_init_bg=True, seed=seed4
            )

        if isinstance(self.custom_scenario, list):
            scenario = self.np_random.choice(self.custom_scenario)
        else:
            scenario = (
                RandomScenario(start_time=start_time, seed=seed3)
                if self.custom_scenario is None
                else self.custom_scenario
            )

        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=seed2)
        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        env = _T1DSimEnv(patient, sensor, pump, scenario)
        return env, seed2, seed3, seed4

    def _render(self, mode="human", close=False):
        self.env.render(close=close)

    def _close(self):
        super()._close()
        self.env._close_viewer()

    @property
    def action_space(self):
        ubasal = self.env.pump._params["max_basal"]
        ubolus = self.env.pump._params["max_bolus"]
        uexercise = self.max_exercise_intensity # max in terms of %VO2 max
        return spaces.Box(low=(0, 0, 0), high=(ubasal, ubolus, uexercise), shape=(3,), dtype=np.float32)

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=1000, shape=(1,))

    @property
    def max_basal(self):
        return self.env.pump._params["max_basal"]
    
    @property   
    def max_bolus(self):
        return self.env.pump._params["max_bolus"]

    @property
    def max_exercise_intensity(self):
        return 1.0 # max in terms of %VO2 max

class BasalBolusT1DSimGymnasiumEnv(gymnasium.Env):
    """
    A custom gymnasium environment for simulating T1D glucose control.
    It provides:
    * a two-dimensional (basal, bolus) action space for insulin control
    * a one-dimensional observation space for CGM readings
    * support for custom reward functions
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 60}
    MAX_BG = 1000

    def __init__(
        self,
        patient_name=None,
        custom_scenario=None,
        reward_fun=None,
        seed=None,
        render_mode=None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.env = BasalBolusT1DSimEnv(
            patient_name=patient_name,
            custom_scenario=custom_scenario,
            reward_fun=reward_fun,
            seed=seed,
        )
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=self.MAX_BG, shape=(1,), dtype=np.float32
        )
        self.action_space = gymnasium.spaces.Box(
            low=np.array([0, 0, 0], dtype=np.float32),
            high=np.array([self.env.max_basal, self.env.max_bolus, self.env.max_exercise_intensity], dtype=np.float32),
            shape=(3,),
            dtype=np.float32
        )

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # Truncated will be controlled by TimeLimit wrapper when registering the env.
        # For example,
        # register(
        #     id="simglucose/adolescent2-v0",
        #     entry_point="simglucose.envs:T1DSimGymnaisumEnv",
        #     max_episode_steps=10,
        #     kwargs={"patient_name": "adolescent#002"},
        # )
        # Once the max_episode_steps is set, the truncated value will be overridden.
        truncated = False
        return np.array([obs.CGM], dtype=np.int32), reward, done, truncated, info


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs, _, _, info = self.env._raw_reset()
        return np.array([obs.CGM], dtype=np.float32), info

    def render(self):
        if self.render_mode == "human":
            self.env.render()

    def close(self):
        self.env.close()

class DiscBasalBolusT1DSimEnv(gym.Env):
    """
    A discretized wrapper of simglucose.simulation.env.T1DSimEnv to support gym API with basal/bolus action space.
    
    States and actions are the same as BasalBolusT1DSimEnv, but the states and actions are discretized into bins.
    The number of bins can be controlled by changing the action_space to spaces.Discrete(n_bins) and modifying the _step function to map the discrete action to a continuous action.
    """

    metadata = {"render.modes": ["human"]}

    SENSOR_HARDWARE = "Dexcom"
    INSULIN_PUMP_HARDWARE = "Insulet"

    def __init__(
        self, patient_name=None, custom_scenario=None, reward_fun=None, seed=None,
        n_bins_basal=100, n_bins_bolus=10, n_bins_exercise=5
    ):
        """
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        """
        # have to hard code the patient_name, gym has some interesting
        # error when choosing the patient
        if patient_name is None:
            patient_name = ["adolescent#001"]

        self.patient_name = patient_name
        self.reward_fun = reward_fun
        self.np_random, _ = seeding.np_random(seed=seed)
        self.custom_scenario = custom_scenario
        self.env, _, _, _ = self._create_env()
        self.n_bins_basal = n_bins_basal
        self.n_bins_bolus = n_bins_bolus
        self.n_bins_exercise = n_bins_exercise
        self.bins_basal = np.linspace(0, self.max_basal, n_bins_basal + 1)
        self.bins_bolus = np.linspace(0, self.max_bolus, n_bins_bolus + 1)
        self.bins_exercise = np.linspace(0, self.max_exercise_intensity, n_bins_exercise + 1)

    def discrete_to_continuous_action(self, discrete_action):
        """
        Maps the discrete action to a continuous action.
        The discrete action is a tuple of (basal_bin, bolus_bin, exercise_bin), where each element is an integer representing the bin index for basal, bolus, and exercise intensity respectively. The continuous action is a tuple of (basal, bolus, exercise_intensity), where each element is a float representing the actual basal insulin, bolus insulin, and exercise intensity.
        
        Takes lower end of bins.
        """
        return self.bins_basal[discrete_action[0]], self.bins_bolus[discrete_action[1]], self.bins_exercise[discrete_action[2]]

    def _step(self, action: float):
        # This gym only controls basal insulin
        c_action = self.discrete_to_continuous_action(action)
        act = Action(basal=c_action[0], bolus=c_action[1], exercise_intensity=c_action[2])
        if self.reward_fun is None:
            return self.env.step(act)
        return self.env.step(act, reward_fun=self.reward_fun)

    def _raw_reset(self):
        return self.env.reset()

    def _reset(self):
        self.env, _, _, _ = self._create_env()
        obs, _, _, _ = self.env.reset()
        return obs

    def _seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        self.env, seed2, seed3, seed4 = self._create_env()
        return [seed1, seed2, seed3, seed4]

    def _create_env(self):
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(self.np_random.randint(0, 1000)) % 2**31
        seed3 = seeding.hash_seed(seed2 + 1) % 2**31
        seed4 = seeding.hash_seed(seed3 + 1) % 2**31

        hour = self.np_random.randint(low=0.0, high=24.0)
        start_time = datetime(2018, 1, 1, hour, 0, 0)

        if isinstance(self.patient_name, list):
            patient_name = self.np_random.choice(self.patient_name)
            patient = T1DPatient.withName(patient_name, random_init_bg=True, seed=seed4)
        else:
            patient = T1DPatient.withName(
                self.patient_name, random_init_bg=True, seed=seed4
            )

        if isinstance(self.custom_scenario, list):
            scenario = self.np_random.choice(self.custom_scenario)
        else:
            scenario = (
                RandomScenario(start_time=start_time, seed=seed3)
                if self.custom_scenario is None
                else self.custom_scenario
            )

        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=seed2)
        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        env = _T1DSimEnv(patient, sensor, pump, scenario)
        return env, seed2, seed3, seed4

    def _render(self, mode="human", close=False):
        self.env.render(close=close)

    def _close(self):
        super()._close()
        self.env._close_viewer()

    @property
    def action_space(self):
        """
        Discretized action space for basal, bolus, and exercise intensity.
        """
        return spaces.MultiDiscrete([self.n_bins_basal, self.n_bins_bolus, self.n_bins_exercise]) # Example: 10 bins for basal, 10 bins for bolus, 5 bins for exercise intensity

    @property
    def observation_space(self):
        """
        Descretized observation space for CGM readings.
        """
        return spaces.Discrete(self.n_bins_cgm)

    @property
    def max_basal(self):
        return self.env.pump._params["max_basal"]
    
    @property   
    def max_bolus(self):
        return self.env.pump._params["max_bolus"]

    @property
    def max_exercise_intensity(self):
        return 1.0 # max in terms of %VO2 max


class DiscBasalBolusT1DSimGymnasiumEnv(gymnasium.Env):
    """
    A custom gymnasium environment for simulating T1D glucose control.
    It provides:
    * a two-dimensional (basal, bolus) action space for insulin control
    * a one-dimensional observation space for CGM readings
    * support for custom reward functions
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 60}
    MAX_BG = 1000

    def __init__(
        self,
        patient_name=None,
        custom_scenario=None,
        reward_fun=None,
        seed=None,
        render_mode=None,
        n_bins_cgm=MAX_BG,
        n_bins_basal=100,
        n_bins_bolus=10,
        n_bins_exercise=5
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.env = DiscBasalBolusT1DSimEnv(
            patient_name=patient_name,
            custom_scenario=custom_scenario,
            reward_fun=reward_fun,
            seed=seed,
            n_bins_basal=n_bins_basal,
            n_bins_bolus=n_bins_bolus,
            n_bins_exercise=n_bins_exercise
        )
        self.n_bins_cgm = n_bins_cgm
        self.observation_space = spaces.Discrete(n_bins_cgm)
        self.action_space = spaces.MultiDiscrete([n_bins_basal, n_bins_bolus, n_bins_exercise]) # Example: 10 bins for basal, 10 bins for bolus, 5 bins for exercise intensity
        self.constraints = set()
        self.valid_actions = set()
        self.add_ground_truth_constraints() # Initialize the set of valid actions based on the ground truth constraints defined in the controller

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # Truncated will be controlled by TimeLimit wrapper when registering the env.
        # For example,
        # register(
        #     id="simglucose/adolescent2-v0",
        #     entry_point="simglucose.envs:T1DSimGymnaisumEnv",
        #     max_episode_steps=10,
        #     kwargs={"patient_name": "adolescent#002"},
        # )
        # Once the max_episode_steps is set, the truncated value will be overridden.
        discretized_cgm = self.discretize_cgm(obs)
        truncated = False
        return np.array([discretized_cgm], dtype=np.int32), reward, done, truncated, info

    def constrained(self, state, action) -> bool:
        # Example constraint: If glucose is above 250 mg/dL, do not allow exercise intensity above 0.5
        if action.exercise_intensity > .9:
            return True
        if action.bolus > 20:
            return True
        if action.basal > 5:
            return True
        return False

    def candidate_constraints(self) -> set:
        # This function generates the set of candidate constrained actions based on the defined constraints in the controller. This is used to populate the valid_actions set for error correction and finding closest valid actions.
        candidate_actions = set()
        for basal_bin in range(self.n_bins_basal):
            for bolus_bin in range(self.n_bins_bolus):
                for exercise_bin in range(self.n_bins_exercise):
                    if self.env.constrained(None, Action(basal=basal_bin, bolus=bolus_bin, exercise_intensity=exercise_bin)):
                        candidate_actions.add((basal_bin, bolus_bin, exercise_bin))
        return candidate_actions
    
        
    def add_constraints(self, constraints: Iterable[tuple[int, int, int]]) -> None:
        # filters particular actions from the action space
        self.constraints = set(constraints)
        self.update_valid_actions()
    
    def relax_constraints(reset=True) -> None:
        if reset == False:
            self.constraints = None
        else:
            self.add_ground_truth_constraints()
            
    @functools.cache
    def add_ground_truth_constraints(self) -> None:
        """
        Iterates over the entire state space (or a representative sample) to determine which actions would be constrained based on the defined logic in the constrained() method. Stores these constraints for use during action selection.
        """
        self.constraints = set()
        # iterates over the full action space using the self.env.action_space and self.env.observation_space to determine which actions are valid or invalid based on the constrained() method. This is a computationally expensive process, so in practice you would want to sample states and actions rather than exhaustively iterate over the entire space.
        for basal_bin in range(len(self.env.unwrapped.bins_basal)):
            for bolus_bin in range(len(self.env.unwrapped.bins_bolus)):
                for exercise_bin in range(len(self.env.unwrapped.bins_exercise)):
                    action = Action(
                        basal=self.env.unwrapped.bins_basal[basal_bin],
                        bolus=self.env.unwrapped.bins_bolus[bolus_bin],
                        exercise_intensity=self.env.unwrapped.bins_exercise[exercise_bin]
                    )
                    # Here we would also need to iterate over a representative sample of states (glucose levels, meal intakes, etc.) to check if the action is constrained in those states. For simplicity, let's assume we are only checking constraints based on the action itself.
                    if self.constrained(None, action): # We can pass None for state if our constraints only depend on the action
                        self.constraints.add((basal_bin, bolus_bin, exercise_bin))
                    else:
                        self.valid_actions.add((basal_bin, bolus_bin, exercise_bin))        
    

    def update_valid_actions(self) -> None:
        # This method can be called to update the set of valid actions if the constraints change over time (e.g., based on changing patient conditions).
        for basal_bin in range(len(self.env.unwrapped.bins_basal)):
            for bolus_bin in range(len(self.env.unwrapped.bins_bolus)):
                for exercise_bin in range(len(self.env.unwrapped.bins_exercise)):
                    action = Action(
                        basal=self.env.unwrapped.bins_basal[basal_bin],
                        bolus=self.env.unwrapped.bins_bolus[bolus_bin],
                        exercise_intensity=self.env.unwrapped.bins_exercise[exercise_bin]
                    )
                    if not self.constrained(None, action):
                        self.valid_actions.add((basal_bin, bolus_bin, exercise_bin))
                    else:
                        self.valid_actions.discard((basal_bin, bolus_bin, exercise_bin))
    

    def discretize_cgm(self, obs):
        # Discretize the CGM reading into bins
        # For example, if we have 10 bins and MAX_BG is 1000, then each bin represents a range of 100 mg/dL
        bin_size = self.MAX_BG / self.n_bins_cgm
        disc_obs = int(obs.CGM // bin_size)
        return disc_obs
    
    def discrete_to_continuous_action(self, discrete_action):
        # Map the discrete action to a continuous action
        return self.env.discrete_to_continuous_action(discrete_action)
       
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs, _, _, info = self.env._raw_reset()
        return np.array([obs.CGM], dtype=np.float32), info

    def render(self):
        if self.render_mode == "human":
            self.env.render()

    def close(self):
        self.env.close()
    
    @property
    def bins_basal(self):
        return self.env.bins_basal

    @property
    def bins_bolus(self):
        return self.env.bins_bolus
    
    @property
    def bins_exercise(self):
        return self.env.bins_exercise
