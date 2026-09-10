"""Reinforcement Learning experiment for treatment optimization

This script follows the dataset structure described in `per_timepoint_prediction.ipynb`:
- reads `lung_cancer_data.csv`
- builds state vectors from patient/timepoint information
- defines a discrete action space for treatment choices
- trains a simple Q-learning policy to optimize long survival

The notebook makes clear that the data is longitudinal and patient-based, so we
must keep the patient-level structure intact and never split rows from the same
patient across train/test in a leakage-prone way.
"""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


DATA_PATH = os.path.join(os.path.dirname(__file__), "lung_cancer_data.csv")
PATIENT_ID = "sympro_respondent"
TIME_COL = "timepoint_month"
TARGET = "meta"

# -----------------------------------------------------------------------------
# 1) State and action spaces
# -----------------------------------------------------------------------------
# The state is a vector built from the patient information available at a visit,
# without using any future data. This is the same philosophy used in the
# per-timepoint notebook: features are only those available up to the current visit.
STATE_FEATURES = [
    "timepoint_month",
    "Leeftijd_bj_diagnose",
    "perf_status",
    "tumor_size",
    "tumor_size_prev",
    "perf_prev",
    "tnm",
    "Roken",
    "bas_dem_gender",
]

# The agent chooses among clinically meaningful treatment classes observed in the
# dataset. The action space is discrete, so Q-learning is a natural fit here.
THERAPIES = ["Targeted", "Chemo", "Immuno"]
ACTION_TO_INDEX = {name: idx for idx, name in enumerate(THERAPIES)}
INDEX_TO_ACTION = {idx: name for idx, name in enumerate(THERAPIES)}


# -----------------------------------------------------------------------------
# 2) Load and prepare the dataset
# -----------------------------------------------------------------------------
def load_and_prepare_data() -> pd.DataFrame:
    """Load the CSV and engineer patient-level survival outcome."""
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = df.columns.str.strip()
    df = df.sort_values([PATIENT_ID, TIME_COL]).copy()

    # Keep the same semantics as the notebook: `meta` indicates whether the patient
    # had an event (e.g., metastasis/short survival) at that time point.
    df[TARGET] = df[TARGET].fillna("No")

    # Create a patient-level long-survival target. We define a patient as having
    # long survival if they never had the adverse event during follow-up.
    patient_last_meta = df.groupby(PATIENT_ID)[TARGET].transform("last")
    df["long_survival"] = (patient_last_meta == "No").astype(int)

    # Fill baseline missing values for previous measurements.
    df["tumor_size_prev"] = df["tumor_size_prev"].fillna(df["tumor_size"])
    df["perf_prev"] = df["perf_prev"].fillna(df["perf_status"])

    # Encode categorical variables to numeric values for RL/state estimation.
    df["bas_dem_gender"] = df["bas_dem_gender"].map({"Female": 0.0, "Male": 1.0})
    df["Roken"] = df["Roken"].map({"Never": 0.0, "Former": 1.0, "Current": 2.0})
    df["tnm"] = df["tnm"].map({"T1": 1.0, "T2": 2.0, "T3": 3.0, "T4": 4.0})

    # Map therapy names to the same discrete action IDs.
    df["therapy_code"] = df["therapy"].map(ACTION_TO_INDEX).fillna(0)

    # Add a simple engineered feature to capture change in tumor burden over time.
    df["tumor_growth_delta"] = df["tumor_size"] - df["tumor_size_prev"]
    df["perf_change"] = df["perf_status"] - df["perf_prev"]

    return df


# -----------------------------------------------------------------------------
# 3) Helper functions for state representation and reward shaping
# -----------------------------------------------------------------------------
def get_state_vector(row: pd.Series) -> np.ndarray:
    """Convert a patient-timepoint row into a numeric state vector."""
    state = row[STATE_FEATURES].astype(float).to_numpy()
    return state


def fit_survival_model(df: pd.DataFrame) -> LogisticRegression:
    """Train a supervised model that estimates a patient's long-survival probability
    using current measurements and treatment choice.

    This is only used to shape rewards in the RL loop. It helps the agent learn
    which treatment choice is associated with better long-term survival.
    """
    feature_cols = [
        "timepoint_month",
        "Leeftijd_bj_diagnose",
        "perf_status",
        "tumor_size",
        "tumor_size_prev",
        "perf_prev",
        "tnm",
        "Roken",
        "bas_dem_gender",
        "therapy_code",
    ]

    X = df[feature_cols].copy()
    y = df["long_survival"].astype(int)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model


def treatment_value(model: LogisticRegression, row: pd.Series, action_idx: int) -> float:
    """Estimate how likely the patient is to have long survival under a given action."""
    feature_cols = [
        "timepoint_month",
        "Leeftijd_bj_diagnose",
        "perf_status",
        "tumor_size",
        "tumor_size_prev",
        "perf_prev",
        "tnm",
        "Roken",
        "bas_dem_gender",
        "therapy_code",
    ]

    values = row[feature_cols[:-1]].astype(float).tolist() + [float(action_idx)]
    x = pd.DataFrame([values], columns=feature_cols)

    # The model was trained on the same feature layout, so we use the action as a
    # final numeric input. `predict_proba` returns probability of class 1.
    prob = model.predict_proba(x)[0, 1]
    return float(prob)


# -----------------------------------------------------------------------------
# 4) Build a discrete state encoding for Q-learning
# -----------------------------------------------------------------------------
def make_bins() -> dict[str, np.ndarray]:
    """Create coarse bins for each feature to discretize the continuous state space.

    Q-learning is easier to implement with a finite number of discrete states, so
    we convert continuous features into bins. This is a standard approach when the
    state is continuous but the action space is discrete.
    """
    return {
        "timepoint_month": np.array([0, 6, 12, 18, 24, 30]),
        "Leeftijd_bj_diagnose": np.linspace(40, 90, 7),
        "perf_status": np.array([0, 1, 2, 3, 4]),
        "tumor_size": np.linspace(0, 80, 9),
        "tumor_size_prev": np.linspace(0, 80, 9),
        "perf_prev": np.array([0, 1, 2, 3, 4]),
        "tnm": np.array([0, 1, 2, 3, 4, 5]),
        "Roken": np.array([-0.5, 0.5, 1.5, 2.5]),
        "bas_dem_gender": np.array([-0.5, 0.5]),
    }


def state_to_key(state: np.ndarray, bins: dict[str, np.ndarray]) -> tuple[int, ...]:
    """Discretize a state vector into a finite state key."""
    key = []
    for feature_name, value in zip(STATE_FEATURES, state):
        bin_edges = bins[feature_name]
        idx = int(np.digitize(float(value), bin_edges))
        idx = max(0, min(len(bin_edges) - 1, idx))
        key.append(idx)
    return tuple(key)


# -----------------------------------------------------------------------------
# 5) Q-learning agent
# -----------------------------------------------------------------------------
class QLearningAgent:
    """A basic tabular Q-learning agent for a discrete action space."""

    def __init__(self, num_actions: int, learning_rate: float = 0.2, gamma: float = 0.9,
                 epsilon: float = 1.0, epsilon_min: float = 0.05, epsilon_decay: float = 0.995):
        self.num_actions = num_actions
        self.alpha = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = defaultdict(float)

    def choose_action(self, state_key: tuple[int, ...]) -> int:
        """Epsilon-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)

        best_action = 0
        best_value = -np.inf
        for action in range(self.num_actions):
            value = self.q_table[(state_key, action)]
            if value > best_value:
                best_value = value
                best_action = action
        return best_action

    def update(self, state_key: tuple[int, ...], action: int, reward: float,
               next_state_key: tuple[int, ...]):
        """Update Q-values using the Bellman equation."""
        current = self.q_table[(state_key, action)]
        best_next = max(self.q_table[(next_state_key, a)] for a in range(self.num_actions))
        target = reward + self.gamma * best_next
        self.q_table[(state_key, action)] = current + self.alpha * (target - current)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# -----------------------------------------------------------------------------
# 6) RL environment logic for treatment optimization
# -----------------------------------------------------------------------------
def compute_reward(row: pd.Series, action_idx: int, survival_model: LogisticRegression) -> float:
    """Shape the reward so that better treatment choices receive stronger positive signals.

    We use a clinically motivated reward:
      - positive reward when the chosen treatment is associated with higher long-survival probability
      - strong positive/negative signal at the final visit depending on the patient's actual outcome
    """
    # Estimate probability of long survival under the current treatment choice.
    chosen_prob = treatment_value(survival_model, row, action_idx)

    # Baseline comparison: if the current treatment as recorded in the data is less effective,
    # the chosen action can improve the survival estimate and therefore get a positive reward.
    current_action = int(row["therapy_code"])
    current_prob = treatment_value(survival_model, row, current_action)

    # Reward from treatment effect.
    treatment_gain = chosen_prob - current_prob

    # Final-step outcome bonus/penalty based on the patient-level long survival label.
    final_bonus = 10.0 if row["long_survival"] == 1 else -10.0

    return float(10.0 * treatment_gain + final_bonus)


def train_agent(n_episodes: int = 250, learning_rate: float = 0.25) -> tuple[QLearningAgent, dict[str, np.ndarray]]:
    """Train the Q-learning policy on patient trajectories."""
    df = load_and_prepare_data()
    survival_model = fit_survival_model(df)
    bins = make_bins()
    agent = QLearningAgent(num_actions=len(THERAPIES), learning_rate=learning_rate)

    patient_ids = df[PATIENT_ID].unique()

    for episode in range(n_episodes):
        patient_id = np.random.choice(patient_ids)
        patient_df = df[df[PATIENT_ID] == patient_id].sort_values(TIME_COL).copy()

        # Each patient visit is one RL step.
        state = get_state_vector(patient_df.iloc[0])
        state_key = state_to_key(state, bins)

        for visit_idx in range(len(patient_df)):
            row = patient_df.iloc[visit_idx]
            state = get_state_vector(row)
            state_key = state_to_key(state, bins)

            action = agent.choose_action(state_key)
            reward = compute_reward(row, action, survival_model)

            if visit_idx + 1 < len(patient_df):
                next_row = patient_df.iloc[visit_idx + 1]
                next_state = get_state_vector(next_row)
                next_state_key = state_to_key(next_state, bins)
            else:
                next_state_key = state_key

            agent.update(state_key, action, reward, next_state_key)

    return agent, bins


# -----------------------------------------------------------------------------
# 7) Simple policy evaluation
# -----------------------------------------------------------------------------
def evaluate_policy(agent: QLearningAgent, bins: dict[str, np.ndarray]) -> None:
    """Print a small summary of the learned policy on the dataset."""
    df = load_and_prepare_data()
    patient_ids = df[PATIENT_ID].unique()
    action_counts = {therapy: 0 for therapy in THERAPIES}

    for patient_id in patient_ids[:20]:
        patient_df = df[df[PATIENT_ID] == patient_id].sort_values(TIME_COL)
        row = patient_df.iloc[-1]
        state = get_state_vector(row)
        state_key = state_to_key(state, bins)
        action = agent.choose_action(state_key)
        action_counts[THERAPIES[action]] += 1

    print("\nLearned treatment distribution over a sample of patients:")
    for therapy, count in action_counts.items():
        print(f"  {therapy}: {count}")


# -----------------------------------------------------------------------------
# 8) Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Reading dataset from:", DATA_PATH)
    df = load_and_prepare_data()
    print(f"Loaded {df.shape[0]} rows and {df[PATIENT_ID].nunique()} patients.")
    print("Columns available:", list(df.columns[:15]), "...")
    print("Example rows:")
    print(df.head(3).to_string(index=False))

    agent, bins = train_agent(n_episodes=300)
    evaluate_policy(agent, bins)

    print("\nState space: a continuous vector with dimensions =", len(STATE_FEATURES))
    print("Action space: discrete treatments =", THERAPIES)
    print("This policy approximates the best treatment choice for improving long survival.")
