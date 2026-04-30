from .base import Controller
from .base import ExerciseAction as Action
import numpy as np
import pandas as pd
from importlib import resources
import logging
import functools
from typing import Iterable

logger = logging.getLogger(__name__)
package_path = resources.files('simglucose')
CONTROL_QUEST = str(package_path / "simglucose" / 'params' / 'Quest.csv')
PATIENT_PARA_FILE = str(package_path / "simglucose" / 'params' / 'vpatient_params.csv')

class MOBBExerciseController(Controller):
    def __init__(self, target=140, exercise_threshold=100, preference='neutral'):
        self.quest = pd.read_csv(CONTROL_QUEST)
        self.patient_params = pd.read_csv(PATIENT_PARA_FILE)
        self.target = target
        self.exercise_threshold = exercise_threshold
        self.preference = preference

    def policy(self, observation, reward, done, **kwargs):
        sample_time = kwargs.get('sample_time', 1)
        pname = kwargs.get('patient_name')
        meal = kwargs.get('meal')
        glucose = observation[0]

        # 1. Get standard basal/bolus from base policy
        action = self._bb_policy(pname, meal, glucose, sample_time)
        bolus = None
        # 2. Calculate Base Intensity
        if glucose > self.exercise_threshold:
            # Base intensity scaling (0.0 to 0.8)
            intensity = (glucose - self.exercise_threshold) / (300 - self.exercise_threshold)
        else:
            intensity = 0.0

        # 3. Apply Persona Logic
        # hyper-adverse: Aggressive correction via both insulin and exercise
        if self.preference == 'hyper-adverse':
            bolus = action.bolus * 2.0
            intensity *= 1.5
            
        # hypo-adverse: Very cautious. Reduce bolus and keep exercise intensity low
        elif self.preference == 'hypo-adverse':
            bolus = action.bolus * 0.5
            intensity *= 0.5
            
        # bolus-adverse: Prefers exercise over taking more shots/insulin
        elif self.preference == 'bolus-adverse':
            bolus = action.bolus * 0.3
            intensity *= 2 # Lean harder on exercise to manage glucose
            
        # exercise-adverse: Prefers dosage (insulin) over physical activity
        elif self.preference == 'exercise-adverse':
            bolus = action.bolus * 1.2
            intensity *= 0 # Minimal exercise, rely on the pen/pump
        if bolus is None:
            bolus = action.bolus

        return Action(basal=action.basal, bolus=bolus, exercise_intensity=intensity)

    def _bb_policy(self, name, meal, glucose, env_sample_time):
        if any(self.quest.Name.str.match(name)):
            quest = self.quest[self.quest.Name.str.match(name)]
            params = self.patient_params[self.patient_params.Name.str.match(name)]
            u2ss = params.u2ss.values.item()
            BW = params.BW.values.item()
        else:
            quest = pd.DataFrame([['Average', 1 / 15, 1 / 50, 50, 30]],
                                 columns=['Name', 'CR', 'CF', 'TDI', 'Age'])
            u2ss = 1.43
            BW = 57.0

        basal = u2ss * BW / 6000
        
        if meal > 0:
            bolus = ((meal * env_sample_time) / quest.CR.values + 
                     (glucose > 150) * (glucose - self.target) / quest.CF.values).item()
        else:
            bolus = 0

        return Action(basal=basal, bolus=bolus / env_sample_time, exercise_intensity=0.0)

    def reset(self):
        pass

class DiscretizedMOBBExerciseController(MOBBExerciseController):
    """
    A variant of the BBExerciseController that outputs discretized actions.
    """
    
    def __init__(self, target=140, exercise_threshold=180, env=None, preference='neutral'):
        super().__init__(target, exercise_threshold, preference=preference)
        self.env = env
        # Note: used for first-order delta-sigma modulation. Only use for basal and bolus since these deal with lower values
        self.basal_disc_error = 0.0 # Cumulative error for basal discretization
        self.exercise_disc_error = 0.0 # Cumulative error for exercise intensity discretization
        self.valid_actions = set() # This will be set in add_ground_truth_constraints() to store the set of valid actions after applying constraints
    
    def discretize(self, value:float, bins: np.ndarray) -> np.intp:
        return np.digitize(value, bins) - 1  # np.digitize returns bin index starting from 1, we want it to start from 0 to include no dosage in the first bin
    
    def apply_constraints(self, basal_bin: int, bolus_bin: int, exercise_bin: int, constraints, valid_actions) -> tuple[int, int, int]:
        # checks if the given action (in discretized form) is in the set of constrained actions. If it is, we can either modify the action to the nearest valid action or we can return a default safe action (e.g., no insulin and no exercise). For simplicity, let's return a default safe action if the original action is constrained.
        if (basal_bin, bolus_bin, exercise_bin) in constraints:
            # find closest valid action (this is a simple example, in practice you would want a more sophisticated method to find the closest valid action)
            valid_actions = valid_actions
            if valid_actions:
                closest_action = min(valid_actions, key=lambda x: (x[0] - basal_bin)**2 + (x[1] - bolus_bin)**2 + (x[2] - exercise_bin)**2)
                return closest_action
        else:
            return (basal_bin, bolus_bin, exercise_bin)
    
    def policy(self, observation, reward, done, **kwargs):
        continuous_action = super().policy(observation, reward, done, **kwargs)
        
        # Discretize basal and bolus
        basal_bin = self.discretize(continuous_action.basal, self.env.unwrapped.bins_basal)
        bolus_bin = self.discretize(continuous_action.bolus, self.env.unwrapped.bins_bolus)
        exercise_bin = self.discretize(continuous_action.exercise_intensity, self.env.unwrapped.bins_exercise)
        
        basal_error = continuous_action.basal - self.env.unwrapped.bins_basal[basal_bin]
        self.basal_disc_error += basal_error
        if self.basal_disc_error >= self.env.unwrapped.bins_basal[1] / 2.5: # If error exceeds slightly less than half the bin size, we can move up one bin to reduce error
            basal_bin = 1
            self.basal_disc_error = 0.0 # reset error if we have applied error correction
        exercise_error = continuous_action.exercise_intensity - self.env.unwrapped.bins_exercise[exercise_bin]
        self.exercise_disc_error += exercise_error
        if self.exercise_disc_error >= self.env.unwrapped.bins_exercise[1] / 2.5: # If error exceeds slightly less than half the bin size, we can move
            exercise_bin = 1
            self.exercise_disc_error = 0.0 # reset error if we have applied error correction
        
        # finally apply the constraints to enforce adherence to (simulated) constraints
        basal_bin, bolus_bin, exercise_bin = self.apply_constraints(basal_bin, bolus_bin, exercise_bin, self.env.unwrapped.constraints, self.env.unwrapped.valid_actions)
        
        # Return a new Action with discretized values
        return Action(basal=basal_bin, bolus=bolus_bin, exercise_intensity=exercise_bin)