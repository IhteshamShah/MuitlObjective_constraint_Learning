from .base import Controller
from .base import ExerciseAction as Action
import numpy as np
import pandas as pd
from importlib import resources
import logging

logger = logging.getLogger(__name__)
package_path = resources.files('simglucose')
CONTROL_QUEST = str(package_path / "simglucose" / 'params' / 'Quest.csv')
PATIENT_PARA_FILE = str(package_path / "simglucose" / 'params' / 'vpatient_params.csv')

# Action = namedtuple('ctrller_action', ['basal', 'bolus', 'exercise_intensity'],)

class BBExerciseController(Controller):
    def __init__(self, target=140, exercise_threshold=100):
        self.quest = pd.read_csv(CONTROL_QUEST)
        self.patient_params = pd.read_csv(PATIENT_PARA_FILE)
        self.target = target
        # Threshold above which the patient starts "exercising" to lower BG
        self.exercise_threshold = exercise_threshold 

    def policy(self, observation, reward, done, **kwargs):
        sample_time = kwargs.get('sample_time', 1)
        pname = kwargs.get('patient_name')
        meal = kwargs.get('meal')

        # Get standard insulin action
        action = self._bb_policy(pname, meal, observation, sample_time)
        
        # --- EXERCISE TUNING LOGIC ---
        # Calculate intensity based on glucose excursion.
        # If glucose is high, we ramp up intensity from 0.0 to 0.8 (80% VO2max)
        glucose = observation
        if glucose > self.exercise_threshold:
            # Simple proportional control for intensity
            # Intensity 0.0 at exercise_threshold, scaling to 0.8 at 300 mg/dL
            intensity = (glucose - self.exercise_threshold) / (300 - self.exercise_threshold)
            # apply exercise constraints
            intensity = np.clip(intensity, 0.0, 0.9)
        else:
            intensity = 0.0
            
        # Attach intensity to the action object
        # Note: Ensure your environment/simulator is updated to read action.intensity
        action = Action(basal=action.basal, bolus=action.bolus, exercise_intensity=intensity)
        print(f"BB policy: basal = {action.basal}, bolus = {action.bolus}, and exercise intensity = {action.exercise_intensity}")
        return action

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
        
        # Standard Bolus logic
        if meal > 0:
            bolus = (
                (meal * env_sample_time) / quest.CR.values + (glucose > 150) *
                (glucose - self.target) / quest.CF.values).item()
        else:
            bolus = 0

        bolus = bolus / env_sample_time
        
        # We return a standard Action, which we will monkey-patch with intensity in policy()
        return Action(basal=basal, bolus=bolus, exercise_intensity=0.0)

    def reset(self):
        pass
    
class DiscretizedBBExerciseController(BBExerciseController):
    """
    A variant of the BBExerciseController that outputs discretized actions.
    """
    
    def __init__(self, target=140, exercise_threshold=180, env=None):
        super().__init__(target, exercise_threshold)
        self.env = env
        # Note: used for first-order delta-sigma modulation. Only use for basal and bolus since these deal with lower values
        self.basal_disc_error = 0.0 # Cumulative error for basal discretization
        self.exercise_disc_error = 0.0 # Cumulative error for exercise intensity discretization
    
    def discretize(self, value:float, bins: np.ndarray) -> np.intp:
        return np.digitize(value, bins) - 1  # np.digitize returns bin index starting from 1, we want it to start from 0 to include no dosage in the first bin
    
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
        
        # Return a new Action with discretized values
        return Action(basal=basal_bin, bolus=bolus_bin, exercise_intensity=exercise_bin)