"""Builds a clinical MDP from a longitudinal lung-cancer dataset by discretizing
patient visits into states, mapping recorded therapies to actions, and estimating
action-conditioned transitions and QoL/survival feature maps. Runs MOCI-IRL (from
the local MOCI_IRL.py) over K=2 preference clusters with bootstrap-restart
ensembling to learn reward weights and state-action treatment constraints.
Compares inferred constraints against clinical eligibility masks to compute
TP/FP/eligibility detection metrics, including per-action and threshold-sweep
analyses, and saves the learned preferences, constraints, metrics, and a JSON
summary to disk.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import MOCI_IRL as moci

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import MOCI_IRL as moci
from MOCI_IRL import SearchOptions


DATA_PATH = Path(__file__).with_name("lung_cancer_cirl_dataset_realistic_nice.csv")
RESULTS_DIR = ROOT / "Results" / "results_lung_cancer"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

THERAPY_TO_ACTION = {
    "Targeted": 0,
    "Immuno": 1,
    "Chemo": 2,
    "Surgery": 3,
    "Observation": 2,
    "Supportive": 2,
}
ACTION_NAMES = ["Targeted", "Immuno", "Chemo", "Surgery"]
FEATURE_LABELS = ["QOL", "Survival"]
ACTION_MASK_COLUMNS = ["action_mask_targeted", "action_mask_immuno", "action_mask_chemo", "action_mask_surgery"]

TRAIN_MAX_PATIENTS = 100
NUM_RESTARTS = 6
EM_MAX_ITERS = 12
DKL_THRESHOLD = 0.02
LEARNED_CONSTRAINT_THRESHOLD = 0.125
USE_ACTION_SPECIFIC_THRESHOLDS = True
SEED_BASE = 17
CANDIDATE_SUBSET_SIZE = None
BOOTSTRAP_RESTARTS = True

LAMBDA_PENALTY = "bic"

USE_SAFETY_PREFILTER = True

SEARCH_MODE = "frozen"
LIKELIHOOD = "local"
WEIGHT_BOUND = None
REFIT_ITERS = 3
CANDIDATE_EPSILON = None
CANDIDATE_MAX_COUNT = 0
VIOLATION_NOISE = None
RANK_BY_DISCREPANCY = False
GENERALIZE_CONSTRAINTS = "rule"
RUN_DIAGNOSTICS = True
LAMBDA_SWEEP_GRID = [1e-5, 0.05, 0.25, 0.5, 1.0, 1.7, 3.0, 5.0, 10.0]
SHADOW_MAX_ROUNDS = 80

ACTION_SIGNATURE_FIELDS = {
    0: (6, 7, 8),
    1: (5, 8),
    2: (1, 8),
    3: (0, 1, 4, 8),
}

SAFETY_THRESHOLDS = {
    "max_toxicity_grade": 3,
    "min_egfr_ml_min": 50.0,
    "min_fev1_chemo_pct": 45.0,
    "min_fev1_surgery_pct": 50.0,
    "max_ecog": 2,
    "min_pdl1_pct": 1.0,
}

ACTION_MIN_RECALL = {
    0: 0.08,
    1: 0.08,
    2: 0.03,
    3: 0.06,
}
ACTION_MAX_FPR = {
    0: 0.18,
    1: 0.18,
    2: 0.12,
    3: 0.18,
}


@dataclass
class ClinicalMDP:
    states: list[tuple]
    start_state: int
    goal_state: int
    horizon: int
    transitions: np.ndarray
    feature_map: np.ndarray
    num_features: int
    num_actions: int
    num_states: int
    safety_screen: set[tuple[int, int]] = field(default_factory=set)


def _safe_numeric(series: pd.Series, fill_value: float = 0.0) -> pd.Series:
    """Params: series, fill_value (used for non-numeric/NaN entries). Returns: numeric pd.Series."""
    return pd.to_numeric(series, errors="coerce").fillna(fill_value)


def load_and_prepare_dataset() -> pd.DataFrame:
    """Params: none. Returns: the loaded and preprocessed lung-cancer DataFrame."""
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    df = df.sort_values(["sympro_respondent", "timepoint_month"]).copy()

    for col in ["QoL", "survival_time_months", "tumor_size", "perf_status", "tumor_size_prev", "perf_prev"]:
        if col in df.columns:
            df[col] = _safe_numeric(df[col])

    if "meta" in df.columns:
        df["meta"] = df["meta"].fillna("No").str.strip()
    if "survival_status" in df.columns:
        df["survival_status"] = df["survival_status"].fillna("Dead")

    df["therapy"] = df["therapy"].fillna("Observation")
    df["therapy_code"] = df["therapy"].map(THERAPY_TO_ACTION).fillna(THERAPY_TO_ACTION["Observation"])

    df["tumor_size_prev"] = df["tumor_size_prev"].fillna(df["tumor_size"])
    df["perf_prev"] = df["perf_prev"].fillna(df["perf_status"])

    if "event_indicator" in df.columns:
        df["event_indicator"] = _safe_numeric(df["event_indicator"], 0.0)
    else:
        df["event_indicator"] = np.where(df.get("meta", "No").astype(str).str.lower().eq("yes"), 1.0, 0.0)

    df["survival_score"] = _safe_numeric(df.get("survival_time_months", pd.Series(np.zeros(len(df)))), 0.0)
    max_survival = max(1.0, df["survival_score"].max())
    df["survival_score_norm"] = df["survival_score"] / max_survival

    df["patient_alive"] = df.groupby("sympro_respondent")["survival_status"].transform(lambda s: s.iloc[-1] == "Alive")
    df["patient_survival_rank"] = df.groupby("sympro_respondent")["survival_score"].transform("max")
    df["patient_qol_rank"] = df.groupby("sympro_respondent")["QoL"].transform("max")

    return df


def state_key_for_row(row: pd.Series) -> tuple[int, int, int, int, int, int, int, int, int]:
    """Params: row (a patient-timepoint record). Returns: discretized state-key tuple."""
    month = int(row.get("timepoint_month", 0) or 0)
    perf = float(row.get("perf_status", 0.0) or 0.0)
    tumor = float(row.get("tumor_size", 0.0) or 0.0)
    qol = float(row.get("QoL", 0.0) or 0.0)
    risk = float(row.get("event_indicator", 0.0) or 0.0)
    pdl1 = float(row.get("pdl1_expression_pct", 0.0) or 0.0)
    toxicity = float(row.get("ctcae_toxicity_grade", 0.0) or 0.0)
    egfr_mut = int(row.get("egfr_mutation", 0) or 0)
    alk = int(row.get("alk_translocation", 0) or 0)
    perf_bins = np.array([0.5, 1.5, 2.5, 3.5])
    tumor_bins = np.array([15.0, 35.0, 55.0, 75.0])
    qol_bins = np.array([20.0, 40.0, 60.0, 80.0])
    risk_bins = np.array([0.5])
    pdl1_bins = np.array([0.5, 5.0, 20.0, 50.0])
    tox_bins = np.array([1.5, 2.5, 3.5])

    perf_idx = int(np.digitize(perf, perf_bins))
    tumor_idx = int(np.digitize(tumor, tumor_bins))
    qol_idx = int(np.digitize(qol, qol_bins))
    risk_idx = int(np.digitize(risk, risk_bins))
    pdl1_idx = int(np.digitize(pdl1, pdl1_bins))
    tox_idx = int(np.digitize(toxicity, tox_bins))

    return (
        month,
        perf_idx,
        tumor_idx,
        qol_idx,
        risk_idx,
        pdl1_idx,
        egfr_mut,
        alk,
        tox_idx,
    )


def build_patient_trajectories(
    df: pd.DataFrame,
    max_patients: int | None = None,
) -> tuple[
    list[list[tuple[int, int]]],
    dict[tuple[int, int, int, int, int, int, int, int, int], int],
    dict[int, dict[str, list[float]]],
    dict[tuple[int, int], list[int | None]],
    dict[int, list[int]],
    dict[tuple[int, int], dict[str, list[float]]],
]:
    """Params: df, max_patients (optional cap on number of patients).
    Returns: (trajectories, state_to_idx, state_stats, transition_records,
    start_action_records, state_action_stats).
    """
    patient_groups = []
    for _, patient_df in df.groupby("sympro_respondent", sort=True):
        patient_groups.append(patient_df.sort_values("timepoint_month"))

    if max_patients is not None:
        patient_groups = patient_groups[:max_patients]

    state_to_idx: dict[tuple[int, int, int, int, int, int, int, int, int], int] = {}
    trajectories: list[list[tuple[int, int]]] = []
    state_stats: dict[int, dict[str, list[float]]] = {}
    transition_records: dict[tuple[int, int], list[int | None]] = {}
    start_action_records: dict[int, list[int]] = {}
    state_action_stats: dict[tuple[int, int], dict[str, list[float]]] = {}

    for patient_df in patient_groups:
        step_state_ids: list[int] = []
        step_actions: list[int] = []

        for _, row in patient_df.iterrows():
            key = state_key_for_row(row)
            if key not in state_to_idx:
                state_to_idx[key] = len(state_to_idx) + 1
            state_id = state_to_idx[key]
            step_state_ids.append(state_id)

            action = int(row.get("therapy_code", THERAPY_TO_ACTION["Observation"]))
            action = max(0, min(3, action))
            step_actions.append(action)

            state_stats.setdefault(state_id, {"QoL": [], "Survival": []})
            state_stats[state_id]["QoL"].append(float(row.get("QoL", 0.0) or 0.0))
            state_stats[state_id]["Survival"].append(float(row.get("survival_score_norm", 0.0) or 0.0))

            sa = (state_id, action)
            state_action_stats.setdefault(sa, {"QoL": [], "Survival": []})
            state_action_stats[sa]["QoL"].append(float(row.get("QoL", 0.0) or 0.0))
            state_action_stats[sa]["Survival"].append(float(row.get("survival_score_norm", 0.0) or 0.0))

        if step_state_ids:
            demo = [(0, step_actions[0])]
            demo.extend((sid, act) for sid, act in zip(step_state_ids, step_actions))
            demo.append((-1, -1))
            trajectories.append(demo)

        if step_state_ids:
            start_action_records.setdefault(step_actions[0], []).append(step_state_ids[0])

        for i, s in enumerate(step_state_ids):
            a = step_actions[i]
            sn = step_state_ids[i + 1] if i + 1 < len(step_state_ids) else None
            transition_records.setdefault((s, a), []).append(sn)

    if not trajectories:
        raise RuntimeError("No patient trajectories were generated from the dataset.")

    goal_id = len(state_to_idx) + 1
    trajectories = [demo[:-1] + [(goal_id, -1)] for demo in trajectories]

    return trajectories, state_to_idx, state_stats, transition_records, start_action_records, state_action_stats


def safety_breached_actions(row: pd.Series) -> set[int]:
    """Params: row (a patient-timepoint record). Returns: set of action indices whose safety thresholds are breached."""
    th: dict[str, float] = SAFETY_THRESHOLDS

    def num(col: str) -> float:
        """Params: col (column name). Returns: float value of row[col], or nan if not parseable."""
        try:
            return float(row.get(col))
        except (TypeError, ValueError):
            return float("nan")

    toxic = num("ctcae_toxicity_grade") >= th["max_toxicity_grade"]
    poor_perf = num("perf_status") > th["max_ecog"]
    low_renal = num("egfr_lab_ml_min") < th["min_egfr_ml_min"]
    no_driver = not (num("egfr_mutation") == 1 or num("alk_translocation") == 1)
    low_pdl1 = num("pdl1_expression_pct") < th["min_pdl1_pct"]

    breached: set[int] = set()
    if toxic or no_driver:
        breached.add(0)
    if toxic or low_pdl1:
        breached.add(1)
    if toxic or low_renal or poor_perf or num("fev1_pct_predicted") < th["min_fev1_chemo_pct"]:
        breached.add(2)
    if (
        toxic
        or poor_perf
        or num("fev1_pct_predicted") < th["min_fev1_surgery_pct"]
        or num("timepoint_month") > 0
        or num("event_indicator") >= 1.0
    ):
        breached.add(3)
    return breached


def build_safety_screen(
    df: pd.DataFrame,
    state_to_idx: dict[tuple[int, int, int, int, int, int, int, int, int], int],
) -> set[tuple[int, int]]:
    """Params: df, state_to_idx (state-key to state-id map). Returns: set of (state_id, action) pairs that breach safety thresholds."""
    screen: set[tuple[int, int]] = set()
    for _, row in df.iterrows():
        state_id = state_to_idx[state_key_for_row(row)]
        screen.update((state_id, a) for a in safety_breached_actions(row))
    return screen


def build_clinical_mdp(df: pd.DataFrame, max_patients: int | None = None) -> tuple[ClinicalMDP, list[list[tuple[int, int]]]]:
    """Params: df, max_patients (optional cap on number of patients). Returns: (ClinicalMDP, trajectories)."""
    trajectories, state_to_idx, state_stats, transition_records, start_action_records, state_action_stats = build_patient_trajectories(
        df, max_patients=max_patients
    )

    start_state = 0
    goal_state = len(state_to_idx) + 1
    num_states = len(state_to_idx) + 2
    num_actions = 4
    horizon = max(10, max(len(traj) for traj in trajectories))

    transitions = np.full((num_states, num_actions), -1, dtype=int)
    feature_map = np.zeros((num_states, num_actions, 2), dtype=float)

    for (s, a), next_states in transition_records.items():
        mapped_next = [goal_state if sn is None else int(sn) for sn in next_states]
        values, counts = np.unique(np.array(mapped_next, dtype=int), return_counts=True)
        transitions[int(s), int(a)] = int(values[np.argmax(counts)])

    for s in range(1, num_states - 1):
        observed_next = transitions[s, transitions[s] != -1]
        default_next = int(observed_next[0]) if observed_next.size else goal_state
        for a in range(num_actions):
            if transitions[s, a] == -1:
                transitions[s, a] = default_next

    for a in range(num_actions):
        if a in start_action_records and start_action_records[a]:
            vals, cnts = np.unique(np.array(start_action_records[a], dtype=int), return_counts=True)
            transitions[start_state, a] = int(vals[np.argmax(cnts)])
        else:
            transitions[start_state, a] = 1 if num_states > 2 else goal_state
        transitions[goal_state, a] = goal_state

    states_ordered = list(state_to_idx.keys())
    for state_id in range(1, num_states - 1):
        state_key = states_ordered[state_id - 1]
        month = int(state_key[0])
        perf_idx = int(state_key[1])
        risk_idx = int(state_key[4])
        pdl1_idx = int(state_key[5])
        egfr_mut = int(state_key[6])
        alk = int(state_key[7])
        tox_idx = int(state_key[8])
        has_driver = (egfr_mut == 1) or (alk == 1)
        high_toxicity = tox_idx >= 2
        poor_perf = perf_idx >= 3
        incompatible_targeted = (not has_driver) or high_toxicity or poor_perf
        incompatible_immuno = (pdl1_idx == 0) or high_toxicity or poor_perf
        incompatible_chemo = high_toxicity or poor_perf
        incompatible_surgery = (month > 0) or high_toxicity or poor_perf or (risk_idx >= 1)

        incompatibility_by_action = {
            0: incompatible_targeted,
            1: incompatible_immuno,
            2: incompatible_chemo,
            3: incompatible_surgery,
        }
        for action, incompatible in incompatibility_by_action.items():
            if (state_id, action) not in transition_records and incompatible:
                transitions[state_id, action] = goal_state

    transitions[transitions == -1] = goal_state

    for state_id in range(1, num_states - 1):
        stats = state_stats.get(state_id, {"QoL": [0.0], "Survival": [0.0]})
        qol_mean = float(np.mean(stats["QoL"])) if stats["QoL"] else 0.0
        surv_mean = float(np.mean(stats["Survival"])) if stats["Survival"] else 0.0
        state_key = states_ordered[state_id - 1]
        month = int(state_key[0])
        perf_idx = int(state_key[1])
        risk_idx = int(state_key[4])
        pdl1_idx = int(state_key[5])
        egfr_mut = int(state_key[6])
        alk = int(state_key[7])
        tox_idx = int(state_key[8])

        for action in range(num_actions):
            sa_stats = state_action_stats.get((state_id, action))
            if sa_stats and sa_stats["QoL"]:
                q = float(np.mean(sa_stats["QoL"])) / 100.0
                s = float(np.mean(sa_stats["Survival"]))
            else:
                q = qol_mean / 100.0
                s = surv_mean
                has_driver = (egfr_mut == 1) or (alk == 1)
                high_toxicity = tox_idx >= 2
                poor_perf = perf_idx >= 3
                incompatibility = 0.0

                if action == 0:
                    if not has_driver:
                        incompatibility += 0.20
                    if high_toxicity:
                        incompatibility += 0.12
                    if poor_perf:
                        incompatibility += 0.08
                elif action == 1:
                    if pdl1_idx == 0:
                        incompatibility += 0.20
                    if high_toxicity:
                        incompatibility += 0.10
                    if poor_perf:
                        incompatibility += 0.08
                elif action == 2:
                    if high_toxicity:
                        incompatibility += 0.14
                    if poor_perf:
                        incompatibility += 0.12
                elif action == 3:
                    if month > 0:
                        incompatibility += 0.14
                    if high_toxicity:
                        incompatibility += 0.10
                    if poor_perf:
                        incompatibility += 0.10
                    if risk_idx >= 1:
                        incompatibility += 0.06

                if incompatibility > 0.0:
                    q = max(0.0, q - (0.06 + 0.35 * incompatibility))
                    s = max(0.0, s - incompatibility)
            feature_map[state_id, action, 0] = q
            feature_map[state_id, action, 1] = s

    feature_map[start_state, :, :] = 0.0
    feature_map[goal_state, :, :] = 0.0

    mdp = ClinicalMDP(
        states=states_ordered,
        start_state=start_state,
        goal_state=goal_state,
        horizon=horizon,
        transitions=transitions,
        feature_map=feature_map,
        num_features=2,
        num_actions=num_actions,
        num_states=num_states,
    )
    mdp.constraint_mode = "state_action"
    mdp.action_feasible_mask = None

    if max_patients is not None:
        used_ids = sorted(df["sympro_respondent"].unique())[:max_patients]
        df_used = df[df["sympro_respondent"].isin(used_ids)]
    else:
        df_used = df
    mdp.safety_screen = build_safety_screen(df_used, state_to_idx)

    mdp.constraint_signature = {
        (state_id, action): tuple(
            int(min(states_ordered[state_id - 1][i], 1)) if (action == 3 and i == 0) else int(states_ordered[state_id - 1][i])
            for i in ACTION_SIGNATURE_FIELDS[action]
        )
        for state_id in range(1, num_states - 1)
        for action in range(num_actions)
    }

    return mdp, trajectories


def trajectory_from_patient(patient_df: pd.DataFrame, state_to_idx: dict[tuple[int, int, int, int, int, int, int, int, int], int]) -> list[int]:
    """Params: patient_df, state_to_idx (state-key to state-id map, mutated in place). Returns: list of state ids for the patient."""
    traj = [0]
    for _, row in patient_df.sort_values("timepoint_month").iterrows():
        key = state_key_for_row(row)
        if key not in state_to_idx:
            state_to_idx[key] = len(state_to_idx) + 1
        traj.append(state_to_idx[key])
    traj.append(len(state_to_idx) + 1)
    return traj


def action_clinical_reason(row: pd.Series, action_name: str) -> str:
    """Params: row, action_name. Returns: short clinical reason string why the action is infeasible, or "allowed"."""
    toxicity = float(row.get("ctcae_toxicity_grade", 0.0) or 0.0)
    egfr_mut = int(row.get("egfr_mutation", 0) or 0)
    alk = int(row.get("alk_translocation", 0) or 0)
    pdl1 = float(row.get("pdl1_expression_pct", 0.0) or 0.0)
    eGFR = float(row.get("egfr_lab_ml_min", 0.0) or 0.0)
    fev1 = float(row.get("fev1_pct_predicted", 0.0) or 0.0)
    perf = float(row.get("perf_status", 0.0) or 0.0)
    month = int(row.get("timepoint_month", 0) or 0)
    event = float(row.get("event_indicator", 0.0) or 0.0)

    if action_name == "Targeted":
        if toxicity >= 3:
            return "toxicity_grade>=3"
        if not (egfr_mut == 1 or alk == 1):
            return "EGFR/ALK not eligible"
        return "allowed"

    if action_name == "Immuno":
        if toxicity >= 3:
            return "toxicity_grade>=3"
        if pdl1 < 1.0:
            return "PD-L1 < 1%"
        return "allowed"

    if action_name == "Chemo":
        if toxicity >= 3:
            return "toxicity_grade>=3"
        if eGFR < 50:
            return "eGFR < 50"
        if fev1 < 45:
            return "FEV1 < 45%"
        if perf > 2:
            return "ECOG/perf_status > 2"
        return "allowed"

    if action_name == "Surgery":
        if month > 0:
            return "not first-line month"
        if toxicity >= 3:
            return "toxicity_grade>=3"
        if perf > 2:
            return "ECOG/perf_status > 2"
        if fev1 < 50:
            return "FEV1 < 50%"
        if event >= 1.0:
            return "high event risk"
        return "allowed"

    return "allowed"


def _eligibility_from_masks(row: pd.Series, action_name: str) -> tuple[bool, str]:
    """Params: row, action_name. Returns: (eligible flag, reason string)."""
    if action_name == "Targeted":
        ok = bool(row.get("action_mask_targeted", 1) == 1)
        return ok, "allowed" if ok else action_clinical_reason(row, action_name)
    if action_name == "Immuno":
        ok = bool(row.get("action_mask_immuno", 1) == 1)
        return ok, "allowed" if ok else action_clinical_reason(row, action_name)
    if action_name == "Chemo":
        ok = bool(row.get("action_mask_chemo", 1) == 1)
        return ok, "allowed" if ok else action_clinical_reason(row, action_name)
    if action_name == "Surgery":
        ok = bool(row.get("action_mask_surgery", 1) == 1)
        return ok, "allowed" if ok else action_clinical_reason(row, action_name)
    return True, "allowed"


def _is_predicted_infeasible(inferred_constraints: set, state_id: int, action_idx: int) -> bool:
    """Params: inferred_constraints, state_id, action_idx. Returns: True if the state or (state, action) pair is in inferred_constraints."""
    if state_id in inferred_constraints:
        return True
    if (state_id, action_idx) in inferred_constraints:
        return True
    return False


def build_state_action_summary(
    df: pd.DataFrame,
    state_to_idx: dict[tuple[int, int, int, int, int, int, int, int, int], int],
    inferred_constraints: set,
) -> list[dict]:
    """Params: df, state_to_idx, inferred_constraints. Returns: list of per (state, action) summary dicts."""
    summary: dict[tuple[int, str], dict] = {}

    for _, row in df.iterrows():
        key = state_key_for_row(row)
        state_id = state_to_idx[key]
        for action_idx, action_name in enumerate(ACTION_NAMES[:4]):
            eligible, reason = _eligibility_from_masks(row, action_name)
            rec = summary.setdefault(
                (state_id, action_name),
                {
                    "state_id": int(state_id),
                    "action": action_name,
                    "action_idx": int(action_idx),
                    "n_rows": 0,
                    "n_ineligible": 0,
                    "reasons": {},
                },
            )
            rec["n_rows"] += 1
            if not eligible:
                rec["n_ineligible"] += 1
                rec["reasons"][reason] = rec["reasons"].get(reason, 0) + 1

    out = []
    for (_, _), rec in sorted(summary.items(), key=lambda t: (t[1]["state_id"], ACTION_NAMES.index(t[1]["action"]))):
        infeasible_rate = rec["n_ineligible"] / max(1, rec["n_rows"])
        eligibility_infeasible = bool(infeasible_rate >= 0.5)
        inferred_infeasible = _is_predicted_infeasible(inferred_constraints, rec["state_id"], rec["action_idx"])
        clinical_reason = "allowed"
        if rec["reasons"]:
            clinical_reason = max(rec["reasons"], key=rec["reasons"].get)

        out.append(
            {
                "state_id": rec["state_id"],
                "action": rec["action"],
                "action_idx": rec["action_idx"],
                "eligibility_infeasible": eligibility_infeasible,
                "inferred_infeasible": inferred_infeasible,
                "infeasible_rate": float(infeasible_rate),
                "clinical_reason": clinical_reason,
                "match": bool(eligibility_infeasible == inferred_infeasible),
            }
        )

    return out


def compute_tp_fp_metrics(state_action_constraints: list[dict], inferred_constraints: set) -> dict:
    """Params: state_action_constraints, inferred_constraints (unused, kept for signature compatibility). Returns: dict of TP/FP/TN/FN counts and rates."""
    tp = fp = tn = fn = 0
    for row in state_action_constraints:
        y_true = bool(row.get("eligibility_infeasible", False))
        y_pred = bool(row.get("inferred_infeasible", False))

        if y_true and y_pred:
            tp += 1
        elif (not y_true) and y_pred:
            fp += 1
        elif (not y_true) and (not y_pred):
            tn += 1
        else:
            fn += 1

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "precision": float(tp / (tp + fp) if (tp + fp) > 0 else 0.0),
        "n_state_action_pairs": int(tp + fp + tn + fn),
    }


def compute_per_action_metrics(state_action_constraints: list[dict]) -> list[dict]:
    """Params: state_action_constraints. Returns: list of per-action TP/FP/TN/FN metric dicts."""
    metrics = []
    for action_name in ACTION_NAMES[:4]:
        subset = [r for r in state_action_constraints if r.get("action") == action_name]
        tp = fp = tn = fn = 0
        for row in subset:
            y_true = bool(row.get("eligibility_infeasible", False))
            y_pred = bool(row.get("inferred_infeasible", False))
            if y_true and y_pred:
                tp += 1
            elif (not y_true) and y_pred:
                fp += 1
            elif (not y_true) and (not y_pred):
                tn += 1
            else:
                fn += 1

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics.append(
            {
                "action": action_name,
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
                "tpr": float(tpr),
                "fpr": float(fpr),
                "n_pairs": int(len(subset)),
            }
        )

    return metrics


def build_eval_rows_from_probabilities(
    base_rows: list[dict],
    inferred_probability: dict[tuple[int, int], float],
    threshold: float | dict[int, float],
) -> list[dict]:
    """Params: base_rows, inferred_probability, threshold (float or per-action dict). Returns: base_rows augmented with inferred probability/threshold/infeasibility/match fields."""
    rows = []
    for row in base_rows:
        if "state_key" in row:
            key = (str(row["state_key"]), int(row["action_idx"]))
        else:
            key = (int(row["state_id"]), int(row["action_idx"]))
        p = float(inferred_probability.get(key, 0.0))
        th = float(threshold.get(int(row["action_idx"]), 0.5)) if isinstance(threshold, dict) else float(threshold)
        inferred_infeasible = bool(p >= th)
        rec = dict(row)
        rec["inferred_probability"] = p
        rec["threshold_used"] = th
        rec["inferred_infeasible"] = inferred_infeasible
        rec["match"] = bool(rec["eligibility_infeasible"] == inferred_infeasible)
        rows.append(rec)
    return rows


def compute_per_action_thresholds(
    base_rows: list[dict],
    inferred_probability: dict[tuple[int, int], float],
    default_threshold: float,
) -> dict[int, float]:
    """Params: base_rows, inferred_probability, default_threshold (used when an action has no rows). Returns: dict mapping action index to chosen threshold."""
    thresholds: dict[int, float] = {}
    grid = np.linspace(0.02, 1.0, 50)

    for action_idx in range(4):
        subset = [r for r in base_rows if int(r.get("action_idx", -1)) == action_idx]
        if not subset:
            thresholds[action_idx] = float(default_threshold)
            continue

        min_recall = float(ACTION_MIN_RECALL.get(action_idx, 0.05))
        max_fpr = float(ACTION_MAX_FPR.get(action_idx, 0.20))
        rows_by_threshold: list[dict] = []

        for th in grid:
            rows = build_eval_rows_from_probabilities(subset, inferred_probability, float(th))
            m = compute_tp_fp_metrics(rows, set())
            rows_by_threshold.append(
                {
                    "threshold": float(th),
                    "tpr": float(m["tpr"]),
                    "fpr": float(m["fpr"]),
                    "precision": float(m["precision"]),
                }
            )

        feasible = [r for r in rows_by_threshold if r["fpr"] <= max_fpr]
        target = [r for r in feasible if r["tpr"] >= min_recall]

        if target:
            non_dominated = [r for r in target if r["tpr"] >= r["fpr"]]
            chosen_pool = non_dominated if non_dominated else target
        elif feasible:
            chosen_pool = feasible
        else:
            chosen_pool = rows_by_threshold

        best = max(
            chosen_pool,
            key=lambda r: (
                r["tpr"] - r["fpr"],
                r["tpr"],
                r["precision"],
                -r["fpr"],
                r["threshold"],
            ),
        )
        best_th = float(best["threshold"])

        thresholds[action_idx] = best_th

    return thresholds


def compute_threshold_sweep(base_rows: list[dict], inferred_probability: dict[tuple[int, int], float]) -> list[dict]:
    """Params: base_rows, inferred_probability. Returns: list of TP/FP metric dicts across a threshold grid."""
    sweep = []
    for threshold in np.linspace(0.0, 1.0, 21):
        rows = build_eval_rows_from_probabilities(base_rows, inferred_probability, float(threshold))
        m = compute_tp_fp_metrics(rows, set())
        sweep.append(
            {
                "threshold": float(threshold),
                "tpr": float(m["tpr"]),
                "fpr": float(m["fpr"]),
                "tp": int(m["tp"]),
                "fp": int(m["fp"]),
                "tn": int(m["tn"]),
                "fn": int(m["fn"]),
            }
        )
    return sweep


def build_reference_eval_rows(
    df: pd.DataFrame,
    state_to_idx: dict[tuple[int, int, int, int, int, int, int, int, int], int],
) -> list[dict]:
    """Params: df, state_to_idx. Returns: list of per (state-key, action) eligibility summary dicts."""
    summary: dict[tuple[tuple[int, int, int, int, int, int, int, int, int], int], dict] = {}

    for _, row in df.iterrows():
        key = state_key_for_row(row)
        for action_idx, action_name in enumerate(ACTION_NAMES[:4]):
            eligible, reason = _eligibility_from_masks(row, action_name)
            rec = summary.setdefault(
                (key, action_idx),
                {
                    "state_key": key,
                    "state_id": int(state_to_idx[key]),
                    "action": action_name,
                    "action_idx": int(action_idx),
                    "n_rows": 0,
                    "n_ineligible": 0,
                    "reasons": {},
                },
            )
            rec["n_rows"] += 1
            if not eligible:
                rec["n_ineligible"] += 1
                rec["reasons"][reason] = rec["reasons"].get(reason, 0) + 1

    out = []
    for rec in summary.values():
        infeasible_rate = rec["n_ineligible"] / max(1, rec["n_rows"])
        eligibility_infeasible = bool(infeasible_rate >= 0.5)
        clinical_reason = "allowed"
        if rec["reasons"]:
            clinical_reason = max(rec["reasons"], key=rec["reasons"].get)

        out.append(
            {
                "state_id": int(rec["state_id"]),
                "state_key": "|".join(str(v) for v in rec["state_key"]),
                "action": rec["action"],
                "action_idx": rec["action_idx"],
                "eligibility_infeasible": eligibility_infeasible,
                "infeasible_rate": float(infeasible_rate),
                "clinical_reason": clinical_reason,
            }
        )

    out.sort(key=lambda r: (r["state_id"], r["action_idx"]))
    return out


def bootstrap_patient_dataframe(df: pd.DataFrame, sampled_ids: np.ndarray) -> pd.DataFrame:
    """Params: df, sampled_ids (patient ids sampled with replacement). Returns: DataFrame with resampled patients relabeled to distinct ids."""
    groups = {int(pid): g.copy() for pid, g in df.groupby("sympro_respondent", sort=False)}
    chunks = []
    for i, pid in enumerate(sampled_ids.tolist()):
        g = groups[int(pid)].copy()
        g["sympro_respondent"] = int(10_000_000 + i)
        chunks.append(g)
    return pd.concat(chunks, ignore_index=True)


def constraints_to_state_key_pairs(mdp: ClinicalMDP, constraint_set: set) -> set[tuple[str, int]]:
    """Params: mdp, constraint_set (state ids or (state, action) tuples). Returns: set of (state-key string, action) pairs."""
    pairs: set[tuple[str, int]] = set()
    num_core_states = len(mdp.states)

    for c in constraint_set:
        if isinstance(c, tuple) and len(c) == 2:
            s, a = int(c[0]), int(c[1])
            if 1 <= s <= num_core_states and 0 <= a < 4:
                key = "|".join(str(v) for v in mdp.states[s - 1])
                pairs.add((key, a))
        elif isinstance(c, (int, np.integer)):
            s = int(c)
            if 1 <= s <= num_core_states:
                key = "|".join(str(v) for v in mdp.states[s - 1])
                for a in range(4):
                    pairs.add((key, a))

    return pairs


def generate_paper_figures(results_dir: Path | None = None) -> list[str]:
    """Params: results_dir (defaults to RESULTS_DIR). Returns: list of figure file paths written by make_paper_figures.main."""
    import make_paper_figures

    return make_paper_figures.main(RESULTS_DIR if results_dir is None else results_dir)


def _search_options() -> SearchOptions:
    """Params: none. Returns: SearchOptions built from the module-level search configuration constants."""
    return SearchOptions(
        mode=SEARCH_MODE,
        likelihood=LIKELIHOOD,
        weight_bound=WEIGHT_BOUND,
        refit_iters=int(REFIT_ITERS),
        candidate_epsilon=CANDIDATE_EPSILON,
        candidate_max_count=int(CANDIDATE_MAX_COUNT),
        violation_noise=VIOLATION_NOISE,
        rank_by_discrepancy=bool(RANK_BY_DISCREPANCY),
        generalize=GENERALIZE_CONSTRAINTS,
    )


def _state_key_to_id(mdp: ClinicalMDP) -> dict[str, int]:
    """Params: mdp. Returns: dict mapping state-key string to state id."""
    return {"|".join(str(v) for v in key): i + 1 for i, key in enumerate(mdp.states)}


def compute_low_tpr_diagnostics(records: list[dict], base_eval_rows: list[dict]) -> dict:
    """Params: records (per-restart mdp/trajectory/diagnostics records), base_eval_rows. Returns: dict with "per_restart" and "lambda_sweep" diagnostic lists."""
    gt_rows = [r for r in base_eval_rows if r["eligibility_infeasible"]]
    restart_summaries: list[dict] = []
    sweep_rows: list[dict] = []

    for rec in records:
        mdp, diag, pairs = rec["mdp"], rec["diag"], rec["pairs"]
        ids = _state_key_to_id(mdp)
        core_states = set(range(1, len(mdp.states) + 1))
        reach = moci.reachable_states(mdp) & core_states

        base_w, base_p = diag.get("baseline_weights"), diag.get("baseline_priors")
        pi_unc = moci.model_policy(mdp, base_w, base_p, set()) if base_w is not None else None
        sa_counts, _ = moci.empirical_action_stats(mdp, rec["trajectories"])

        per_pair = []
        for r in gt_rows:
            sid = ids.get(str(r["state_key"]))
            if sid is None:
                continue
            a = int(r["action_idx"])
            per_pair.append(
                {
                    "reachable": sid in reach,
                    "detected": (str(r["state_key"]), a) in pairs,
                    "pi_unconstrained": float(pi_unc[sid, a]) if pi_unc is not None else float("nan"),
                    "demonstrated": sa_counts.get((sid, a), 0) > 0,
                }
            )
        gdf = pd.DataFrame(per_pair)

        def mean_of(mask, col):
            """Params: mask (boolean row selector), col (column name). Returns: mean of gdf[col] over masked rows, or nan if empty."""
            sub = gdf[mask]
            return float(sub[col].mean()) if len(sub) else float("nan")

        missed = ~gdf["detected"]
        restart_summaries.append(
            {
                "restart": rec["restart"],
                "n_states": len(core_states),
                "n_states_reachable_from_start": len(reach),
                "n_gt_pairs": int(len(gdf)),
                "frac_gt_pairs_in_reachable_states": float(gdf["reachable"].mean()),
                "tpr": float(gdf["detected"].mean()),
                "tpr_reachable_states": mean_of(gdf["reachable"], "detected"),
                "tpr_unreachable_states": mean_of(~gdf["reachable"], "detected"),
                "frac_gt_pairs_demonstrated_by_expert": float(gdf["demonstrated"].mean()),
                "mean_pi_unconstrained_detected": mean_of(gdf["detected"], "pi_unconstrained"),
                "mean_pi_unconstrained_missed": mean_of(missed, "pi_unconstrained"),
                "frac_missed_absorbed_pi_lt_0.05": float((gdf[missed]["pi_unconstrained"] < 0.05).mean()) if missed.any() else float("nan"),
                "frac_missed_in_unreachable_states": float((~gdf[missed]["reachable"]).mean()) if missed.any() else float("nan"),
            }
        )

        trace = diag.get("trace") if SEARCH_MODE == "frozen" else None
        if trace is not None:
            covered_from = min(float(rec["lambda_search"]), float(diag.get("shadow_floor", np.inf)))
            if diag.get("shadow_capped") and trace:
                covered_from = max(covered_from, float(trace[-1]["delta_logL"]))
            for lam in LAMBDA_SWEEP_GRID:
                if lam < covered_from - 1e-12:
                    continue
                chosen: set = set()
                for entry in trace:
                    if entry["delta_logL"] > lam:
                        chosen.update(tuple(p) for p in entry["pairs"])
                    else:
                        break
                chosen_pairs = constraints_to_state_key_pairs(mdp, chosen)
                prob = {
                    (str(r["state_key"]), int(r["action_idx"])): 1.0 if (str(r["state_key"]), int(r["action_idx"])) in chosen_pairs else 0.0
                    for r in base_eval_rows
                }
                m = compute_tp_fp_metrics(build_eval_rows_from_probabilities(base_eval_rows, prob, 0.5), set())
                sweep_rows.append(
                    {
                        "restart": rec["restart"],
                        "lambda": float(lam),
                        "n_constraints": int(len(chosen)),
                        "tpr": m["tpr"],
                        "fpr": m["fpr"],
                        "precision": m["precision"],
                    }
                )

    return {"per_restart": restart_summaries, "lambda_sweep": sweep_rows}


def run_lung_cancer_moci() -> dict:
    """Params: none. Returns: dict of learned weights, priors, inferred constraints, detection metrics, and training config."""
    df = load_and_prepare_dataset()
    df = df[df.get("QoL").notna()].copy()

    max_patients = min(TRAIN_MAX_PATIENTS, df["sympro_respondent"].nunique()) if TRAIN_MAX_PATIENTS else None
    if max_patients is not None:
        patient_ids_sorted = sorted(df["sympro_respondent"].unique())[:max_patients]
        df_train = df[df["sympro_respondent"].isin(patient_ids_sorted)].copy()
    else:
        df_train = df.copy()

    trajectories_ref, state_to_idx_ref, _, _, _, _ = build_patient_trajectories(df_train, max_patients=None)
    mdp_ref, _ = build_clinical_mdp(df_train, max_patients=None)

    base_eval_rows = build_reference_eval_rows(df_train, state_to_idx_ref)
    ref_patient_ids = np.array(sorted(df_train["sympro_respondent"].unique()), dtype=int)
    search_options = _search_options()

    def make_problem(restart_idx: int, rng: np.random.Generator):
        """Params: restart_idx, rng. Returns: (ClinicalMDP, trajectories) built from a bootstrap resample of the training patients."""
        if BOOTSTRAP_RESTARTS:
            sampled_ids = rng.choice(ref_patient_ids, size=len(ref_patient_ids), replace=True)
            df_restart = bootstrap_patient_dataframe(df_train, sampled_ids)
        else:
            df_restart = df_train.copy()
        return build_clinical_mdp(df_restart, max_patients=None)

    def make_diagnostics(restart_idx: int, mdp: ClinicalMDP) -> dict:
        """Params: restart_idx, mdp. Returns: dict of diagnostics settings, or empty dict if diagnostics are disabled."""
        if RUN_DIAGNOSTICS and SEARCH_MODE == "frozen":
            return {"shadow_floor": float(min(LAMBDA_SWEEP_GRID)), "max_shadow_rounds": int(SHADOW_MAX_ROUNDS)}
        return {}

    ensemble = moci.run_moci_ensemble(
        make_problem,
        K=2,
        n_restarts=NUM_RESTARTS,
        seed_base=SEED_BASE,
        d_DKL=DKL_THRESHOLD,
        max_em_iters=EM_MAX_ITERS,
        candidate_subset_size=CANDIDATE_SUBSET_SIZE,
        lambda_penalty=LAMBDA_PENALTY,
        options=search_options,
        safety_filter_fn=(lambda m: m.safety_screen) if USE_SAFETY_PREFILTER else None,
        canonicalize=constraints_to_state_key_pairs,
        diagnostics_factory=make_diagnostics,
    )

    restart_rows = []
    diagnostic_records: list[dict] = []
    for rec in ensemble.restarts:
        mdp, trajectories = rec["mdp"], rec["demos"]
        inferred_pairs = rec["canonical"]
        diagnostic_records.append(
            {
                "restart": int(rec["restart"]),
                "mdp": mdp,
                "trajectories": trajectories,
                "diag": rec["diagnostics"],
                "pairs": inferred_pairs,
                "lambda_search": moci.resolve_lambda(LAMBDA_PENALTY, len(trajectories), DKL_THRESHOLD),
            }
        )
        restart_prob = {
            (str(r["state_key"]), int(r["action_idx"])): 1.0 if (str(r["state_key"]), int(r["action_idx"])) in inferred_pairs else 0.0
            for r in base_eval_rows
        }
        eval_rows = build_eval_rows_from_probabilities(base_eval_rows, restart_prob, threshold=0.5)
        metrics = compute_tp_fp_metrics(eval_rows, set())
        restart_rows.append(
            {
                "restart": int(rec["restart"]),
                "seed": int(rec["seed"]),
                "bootstrap": bool(BOOTSTRAP_RESTARTS),
                "joint_log_likelihood_avg": rec["log_like"],
                "n_constraints": int(len(rec["constraints"])),
                "tpr": float(metrics["tpr"]),
                "fpr": float(metrics["fpr"]),
                "precision": float(metrics["precision"]),
            }
        )

    inferred_votes = {
        (str(r["state_key"]), int(r["action_idx"])): ensemble.votes.get((str(r["state_key"]), int(r["action_idx"])), 0)
        for r in base_eval_rows
    }
    best = {
        "constraints": ensemble.best["constraints"],
        "weights": ensemble.best["weights"],
        "priors": ensemble.best["priors"],
        "log_like": ensemble.best["log_like"],
        "mdp": ensemble.best["mdp"],
    }

    inferred_probability = {k: v / float(NUM_RESTARTS) for k, v in inferred_votes.items()}
    action_thresholds = None
    threshold_for_eval: float | dict[int, float] = float(LEARNED_CONSTRAINT_THRESHOLD)
    if USE_ACTION_SPECIFIC_THRESHOLDS:
        action_thresholds = compute_per_action_thresholds(
            base_eval_rows,
            inferred_probability,
            default_threshold=float(LEARNED_CONSTRAINT_THRESHOLD),
        )
        threshold_for_eval = action_thresholds

    state_action_constraints = build_eval_rows_from_probabilities(
        base_eval_rows,
        inferred_probability,
        threshold=threshold_for_eval,
    )
    threshold_sweep = compute_threshold_sweep(base_eval_rows, inferred_probability)
    constraint_detection = compute_tp_fp_metrics(state_action_constraints, set())
    per_action_detection = compute_per_action_metrics(state_action_constraints)

    inferred_constraints = best["constraints"]
    final_weights = best["weights"]
    final_priors = best["priors"]
    inferred_constraint_keys = sorted(
        [{"state_key": k, "action_idx": int(a)} for (k, a) in constraints_to_state_key_pairs(best["mdp"], inferred_constraints)],
        key=lambda r: (r["state_key"], r["action_idx"]),
    )

    responsibilities = moci.e_step(
        mdp_ref,
        trajectories_ref,
        set(),
        final_weights,
        np.array(final_priors, dtype=float),
    )
    cluster_ids = np.argmax(responsibilities, axis=1).astype(int)
    ordered_patients = list(df_train.groupby("sympro_respondent", sort=True).groups.keys())[: len(cluster_ids)]
    cluster_map = {int(pid): int(cluster_ids[i]) for i, pid in enumerate(ordered_patients)}
    cluster_probabilities = []
    for i, pid in enumerate(ordered_patients):
        rec = {"sympro_respondent": int(pid), "cluster": int(cluster_ids[i])}
        for k in range(responsibilities.shape[1]):
            rec[f"cluster_{k}_prob"] = float(responsibilities[i, k])
        cluster_probabilities.append(rec)

    constraints = sorted(list(inferred_constraints))
    results = {
        "n_patients": int(df["sympro_respondent"].nunique()),
        "n_patients_used_for_training": int(df_train["sympro_respondent"].nunique()),
        "n_trajectories": len(trajectories_ref),
        "num_states": mdp_ref.num_states,
        "goal_state": mdp_ref.goal_state,
        "start_state": mdp_ref.start_state,
        "constraint_type": "state_action",
        "constraint_definition": "hard treatment constraints learned over (state, action) without pre-imposed masks",
        "inferred_constraints": constraints,
        "inferred_constraint_keys": inferred_constraint_keys,
        "state_action_constraints": state_action_constraints,
        "state_action_summary": [
            {
                "state_id": row["state_id"],
                "action": row["action"],
                "eligibility_infeasible": row["eligibility_infeasible"],
                "inferred_infeasible": row["inferred_infeasible"],
                "infeasible_rate": row["infeasible_rate"],
                "clinical_reason": row["clinical_reason"],
            }
            for row in state_action_constraints
            if row["eligibility_infeasible"] or row["inferred_infeasible"]
        ],
        "final_priors": [float(v) for v in final_priors],
        "weights": [[float(v) for v in w] for w in final_weights],
        "constraint_detection": constraint_detection,
        "per_action_detection": per_action_detection,
        "threshold_sweep": threshold_sweep,
        "restart_metrics": restart_rows,
        "inferred_probability_threshold": float(LEARNED_CONSTRAINT_THRESHOLD),
        "action_specific_thresholds": action_thresholds,
        "cluster_assignments": [{"sympro_respondent": int(pid), "cluster": int(cid)} for pid, cid in cluster_map.items()],
        "cluster_probabilities": cluster_probabilities,
        "feature_labels": FEATURE_LABELS,
        "action_labels": ACTION_NAMES,
        "dataset_note": (
            "Two learned preference clusters correspond to QoL-first and survival-first care "
            "preferences in a longitudinal lung-cancer setting."
        ),
        "training_config": {
            "num_restarts": int(NUM_RESTARTS),
            "bootstrap_restarts": bool(BOOTSTRAP_RESTARTS),
            "em_max_iters": int(EM_MAX_ITERS),
            "d_dkl": float(DKL_THRESHOLD),
            "lambda_penalty": LAMBDA_PENALTY,
            "search_mode": SEARCH_MODE,
            "likelihood": LIKELIHOOD,
            "weight_bound": WEIGHT_BOUND,
            "candidate_epsilon": CANDIDATE_EPSILON,
            "candidate_max_count": int(CANDIDATE_MAX_COUNT),
            "violation_noise": VIOLATION_NOISE,
            "rank_by_discrepancy": bool(RANK_BY_DISCREPANCY),
            "generalize_constraints": GENERALIZE_CONSTRAINTS,
            "use_safety_prefilter": bool(USE_SAFETY_PREFILTER),
            "safety_thresholds": dict(SAFETY_THRESHOLDS),
            "train_max_patients": int(TRAIN_MAX_PATIENTS),
            "seed_base": int(SEED_BASE),
            "candidate_subset_size": None if CANDIDATE_SUBSET_SIZE is None else int(CANDIDATE_SUBSET_SIZE),
            "use_action_specific_thresholds": bool(USE_ACTION_SPECIFIC_THRESHOLDS),
        },
    }

    if RUN_DIAGNOSTICS:
        results["low_tpr_diagnostics"] = compute_low_tpr_diagnostics(diagnostic_records, base_eval_rows)


    return results


def save_results(results: dict) -> None:
    """Params: results (dict produced by run_lung_cancer_moci). Returns: None; writes CSV/JSON outputs to RESULTS_DIR."""
    RESULTS_DIR.mkdir(exist_ok=True)

    weights_path = RESULTS_DIR / "learned_preferences.csv"
    headers = ["cluster", "QoL", "Survival"]
    rows = []
    for idx, weight_vec in enumerate(results["weights"]):
        rows.append({"cluster": idx, "QoL": float(weight_vec[0]), "Survival": float(weight_vec[1])})
    pd.DataFrame(rows, columns=headers).to_csv(weights_path, index=False)

    constraints_path = RESULTS_DIR / "inferred_constraints.csv"
    pd.DataFrame(results.get("state_action_summary", [])).to_csv(constraints_path, index=False)

    compare_path = RESULTS_DIR / "eligibility_vs_learned_constraints.csv"
    pd.DataFrame(results.get("state_action_constraints", [])).to_csv(compare_path, index=False)

    assignments_path = RESULTS_DIR / "cluster_assignments.csv"
    pd.DataFrame(results.get("cluster_assignments", [])).to_csv(assignments_path, index=False)

    detection_path = RESULTS_DIR / "constraint_detection_metrics.csv"
    pd.DataFrame([results.get("constraint_detection", {})]).to_csv(detection_path, index=False)

    per_action_path = RESULTS_DIR / "constraint_detection_per_action.csv"
    pd.DataFrame(results.get("per_action_detection", [])).to_csv(per_action_path, index=False)

    threshold_path = RESULTS_DIR / "constraint_threshold_sweep.csv"
    pd.DataFrame(results.get("threshold_sweep", [])).to_csv(threshold_path, index=False)

    restart_path = RESULTS_DIR / "constraint_restart_metrics.csv"
    pd.DataFrame(results.get("restart_metrics", [])).to_csv(restart_path, index=False)

    diagnostics = results.get("low_tpr_diagnostics")
    if diagnostics:
        pd.DataFrame(diagnostics.get("per_restart", [])).to_csv(RESULTS_DIR / "diagnostics_low_tpr.csv", index=False)
        if diagnostics.get("lambda_sweep"):
            pd.DataFrame(diagnostics["lambda_sweep"]).to_csv(RESULTS_DIR / "lambda_sweep.csv", index=False)

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    print(f"Saved learned preferences to: {weights_path}")
    print(f"Saved inferred constraints to: {constraints_path}")
    print(f"Saved eligibility comparison to: {compare_path}")
    print(f"Saved cluster assignments to: {assignments_path}")
    print(f"Saved TP/FP metrics to: {detection_path}")
    print(f"Saved per-action metrics to: {per_action_path}")
    print(f"Saved threshold sweep to: {threshold_path}")
    print(f"Saved restart metrics to: {restart_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    results = run_lung_cancer_moci()
    save_results(results)
    generate_paper_figures()
    print("MOCI lung-cancer learning completed.")
    print(json.dumps({
        "constraint_type": results["constraint_type"],
        "inferred_constraints": results["inferred_constraints"],
        "state_action_summary": results["state_action_summary"][:10],
        "priors": results["final_priors"],
        "weights": results["weights"],
    }, indent=2))
