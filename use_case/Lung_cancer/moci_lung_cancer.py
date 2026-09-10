"""MOCI-IRL on the lung-cancer C-IRL dataset.

This script learns a two-cluster preference model from longitudinal oncology data.
It follows the conventions from the dataset description file and the clinical C-IRL
setup used for lung cancer treatment planning.

What the code represents
------------------------
State space:
    Each patient-timepoint is represented by a discrete state that combines:
    - baseline patient context: age, sex, smoking status
    - tumor burden: TNM stage, tumor size, metastasis indicator, treatment response
    - biomarker suitability: EGFR mutation, ALK translocation, PD-L1 expression
    - safety and organ reserve: eGFR, FEV1, toxicity grade, ECOG performance status
    - longitudinal context: treatment line and visit month
    - feasibility masks: targeted/immuno/chemo action masks from the data

    These features define a finite clinical state space S. We discretize them into
    bins to fit the generic MOCI-IRL transition model.

Action space:
    The nominal clinical action set is:
        A = {Targeted, Immuno, Chemo}
    The dataset also includes action masks:
        action_mask_targeted, action_mask_immuno, action_mask_chemo
    These masks encode whether each action is clinically feasible for a patient at a
    specific timepoint. In practice, the generic MOCI optimizer expects a fixed 5-slot
    action interface, so we keep the 3 real therapy actions as the primary actions and
    leave the remaining slots as fallback entries. Feasibility is enforced by the state
    masks and by only allowing transitions consistent with the observed feasible actions.

Preferences:
    The learned reward vector is two-dimensional:
        w = [w_QoL, w_survival]
    The two preference clusters represent different clinical priorities:
      - a QoL-first policy, emphasizing quality-of-life preservation
      - a survival-first policy, emphasizing longer survival and disease control
    MOCI-IRL learns these preferences from patient trajectories without hard-coding the
    weights in advance.

Constraints:
    Constraints are modeled as state-dependent feasibility masks. A treatment action is
    considered valid only if its mask is 1 for that patient-state pair. Formally,
        C(s_t, a_t) = 0 if a_t in A(s_t), else 1
    This forces the learned solution to respect physiologic and biomarker limitations
    (toxicity, renal function, lung reserve, targeted therapy eligibility, PD-L1 status,
    and patient-specific eligibility).

How the learning works:
    1. Patient trajectories are built from repeated longitudinal visits.
    2. Each visit is assigned to a discrete clinical state.
    3. A finite MDP is constructed with these states and action-compatible transitions.
    4. MOCI-IRL runs EM over K=2 preference clusters.
    5. The algorithm alternates:
         E-step: assign each patient trajectory to the most likely preference cluster
         M-step: update reward weights and infer hard-state constraints
    6. Final outputs include the learned reward vectors and the inferred constraint states.

What the results show:
    The final summary contains the learned priors and weight vectors. The cluster with the
    larger weight on survival represents a survival-first preference; the cluster with the
    larger QoL alignment represents a QoL-first preference. The report is useful for
    identifying whether the observed lung-cancer treatment trajectories are better explained
    by a survival-oriented policy or a QoL-preserving policy under the feasibility masks.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import MOCI_IRL as moci


DATA_PATH = Path(__file__).with_name("lung_cancer_cirl_dataset_realistic_nice.csv")
if not DATA_PATH.exists():
    DATA_PATH = Path(__file__).with_name("lung_cancer_cirl_dataset_realistic.csv")
if not DATA_PATH.exists():
    DATA_PATH = Path(__file__).with_name("lung_cancer_cirl_dataset.csv")
if not DATA_PATH.exists():
    DATA_PATH = Path(__file__).with_name("lung_cancer_data.csv")
RESULTS_DIR = Path(__file__).with_name("results")
RESULTS_DIR.mkdir(exist_ok=True)

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
PLOT_FILENAMES = {
    "preferences": "figure_1_preference_weights.png",
    "priors": "figure_2_cluster_priors.png",
    "feasibility": "figure_3_action_feasibility.png",
    "longitudinal": "figure_4_longitudinal_outcomes.png",
    "tp_fp": "figure_5_true_false_positive_rates.png",
    "tp_fp_by_action": "figure_6_per_action_tpr_fpr.png",
    "threshold_sweep": "figure_7_threshold_sweep_roc.png",
    "restart_recall_precision": "figure_8_restart_recall_precision.png",
}

TRAIN_MAX_PATIENTS = 100
NUM_RESTARTS = 8
EM_MAX_ITERS = 12
DKL_THRESHOLD = 0.02
LEARNED_CONSTRAINT_THRESHOLD = 0.125
USE_ACTION_SPECIFIC_THRESHOLDS = True
SEED_BASE = 17
CANDIDATE_SUBSET_SIZE = None
BOOTSTRAP_RESTARTS = True

# Hybrid threshold policy: per-action minimum recall targets with FPR caps.
ACTION_MIN_RECALL = {
    0: 0.08,  # Targeted
    1: 0.08,  # Immuno
    2: 0.03,  # Chemo
    3: 0.06,  # Surgery
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


def _safe_numeric(series: pd.Series, fill_value: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(fill_value)


def load_and_prepare_dataset() -> pd.DataFrame:
    """Load the lung-cancer dataset and apply clinically meaningful preprocessing.

    The dataset is longitudinal: each patient appears across multiple timepoints.
    MOCI needs a set of patient trajectories, so we preserve the time ordering and
    enrich the record with patient-level outcome summaries.
    """
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

    # Patient-level conclusion: longer survival and better QoL correspond to more
    # desirable outcomes. This sits naturally with the two preference goals in the prompt.
    df["patient_alive"] = df.groupby("sympro_respondent")["survival_status"].transform(lambda s: s.iloc[-1] == "Alive")
    df["patient_survival_rank"] = df.groupby("sympro_respondent")["survival_score"].transform("max")
    df["patient_qol_rank"] = df.groupby("sympro_respondent")["QoL"].transform("max")

    return df


def state_key_for_row(row: pd.Series) -> tuple[int, int, int, int, int, int, int, int, int]:
    """Convert a clinical timepoint into a discrete state for MOCI.

    The state includes the visit month and the major clinical risk markers.
    Treatment feasibility masks are used only for post-learning evaluation.
    """
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
    list[list[int]],
    dict[tuple[int, int, int, int, int, int, int, int, int], int],
    dict[int, dict[str, list[float]]],
    dict[tuple[int, int], list[int | None]],
    dict[int, list[int]],
    dict[tuple[int, int], dict[str, list[float]]],
]:
    """Create patient trajectories plus action-conditioned transition/feature summaries."""
    patient_groups = []
    for _, patient_df in df.groupby("sympro_respondent", sort=True):
        patient_groups.append(patient_df.sort_values("timepoint_month"))

    if max_patients is not None:
        patient_groups = patient_groups[:max_patients]

    state_to_idx: dict[tuple[int, int, int, int, int, int, int, int, int], int] = {}
    trajectories: list[list[int]] = []
    state_stats: dict[int, dict[str, list[float]]] = {}
    transition_records: dict[tuple[int, int], list[int | None]] = {}
    start_action_records: dict[int, list[int]] = {}
    state_action_stats: dict[tuple[int, int], dict[str, list[float]]] = {}

    for patient_df in patient_groups:
        patient_states: list[int] = [0]
        step_state_ids: list[int] = []
        step_actions: list[int] = []

        for _, row in patient_df.iterrows():
            key = state_key_for_row(row)
            if key not in state_to_idx:
                state_to_idx[key] = len(state_to_idx) + 1
            state_id = state_to_idx[key]
            patient_states.append(state_id)
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

        patient_states.append(len(state_to_idx) + 1)
        trajectories.append(patient_states)

        if step_state_ids:
            start_action_records.setdefault(step_actions[0], []).append(step_state_ids[0])

        for i, s in enumerate(step_state_ids):
            a = step_actions[i]
            sn = step_state_ids[i + 1] if i + 1 < len(step_state_ids) else None
            transition_records.setdefault((s, a), []).append(sn)

    if not trajectories:
        raise RuntimeError("No patient trajectories were generated from the dataset.")

    return trajectories, state_to_idx, state_stats, transition_records, start_action_records, state_action_stats


def build_clinical_mdp(df: pd.DataFrame, max_patients: int | None = None) -> tuple[ClinicalMDP, list[list[int]]]:
    """Build a MOCI-compatible MDP from the lung-cancer observational trajectories."""
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

    # Action-aware transition model: for each (state, action), route to the most
    # frequently observed next state under that specific therapy.
    for (s, a), next_states in transition_records.items():
        mapped_next = [goal_state if sn is None else int(sn) for sn in next_states]
        values, counts = np.unique(np.array(mapped_next, dtype=int), return_counts=True)
        transitions[int(s), int(a)] = int(values[np.argmax(counts)])

    # For unseen actions in a state, back off to the dominant observed next state.
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

    # Do not explicitly impose treatment feasibility before learning.
    # Fill undefined transitions so the transition model remains total.
    transitions[transitions == -1] = goal_state

    # Build action-conditioned feature summaries; this preserves treatment-specific
    # outcome differences needed for learning action-specific constraints.
    for state_id in range(1, num_states - 1):
        stats = state_stats.get(state_id, {"QoL": [0.0], "Survival": [0.0]})
        qol_mean = float(np.mean(stats["QoL"])) if stats["QoL"] else 0.0
        surv_mean = float(np.mean(stats["Survival"])) if stats["Survival"] else 0.0
        # State key layout:
        # (month, perf_idx, tumor_idx, qol_idx, risk_idx, pdl1_idx, egfr_mut, alk, tox_idx)
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

    return mdp, trajectories


def trajectory_from_patient(patient_df: pd.DataFrame, state_to_idx: dict[tuple[int, int, int, int, int, int, int, int, int], int]) -> list[int]:
    traj = [0]
    for _, row in patient_df.sort_values("timepoint_month").iterrows():
        key = state_key_for_row(row)
        if key not in state_to_idx:
            state_to_idx[key] = len(state_to_idx) + 1
        traj.append(state_to_idx[key])
    traj.append(len(state_to_idx) + 1)
    return traj


def action_clinical_reason(row: pd.Series, action_name: str) -> str:
    """Return the clinical reason why an action is infeasible for a row."""
    toxicity = float(row.get("ctcae_toxicity_grade", 0.0) or 0.0)
    egfr_mut = int(row.get("egfr_mutation", 0) or 0)
    alk = int(row.get("alk_translocation", 0) or 0)
    pdl1 = float(row.get("pdl1_expression_pct", 0.0) or 0.0)
    eGFR = float(row.get("egfr_lab_ml_min", 0.0) or 0.0)
    fev1 = float(row.get("fev1_pct_predicted", 0.0) or 0.0)
    perf = float(row.get("perf_status", 0.0) or 0.0)

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

    return "allowed"


def _eligibility_from_masks(row: pd.Series, action_name: str) -> tuple[bool, str]:
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
        return ok, "allowed" if ok else "not surgery candidate"
    return True, "allowed"


def _is_predicted_infeasible(inferred_constraints: set, state_id: int, action_idx: int) -> bool:
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
    """Summarize eligibility labels and learned state-action constraints."""
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
    """Compare learned constraints against eligibility masks and compute TP/FP rates."""
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
    thresholds: dict[int, float] = {}
    # Skip 0.0 to avoid the degenerate "predict everything infeasible" solution.
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
    """Build evaluation rows keyed by clinical state signature, not transient state ids."""
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
    """Create a bootstrap patient cohort where sampled trajectories remain distinct."""
    groups = {int(pid): g.copy() for pid, g in df.groupby("sympro_respondent", sort=False)}
    chunks = []
    for i, pid in enumerate(sampled_ids.tolist()):
        g = groups[int(pid)].copy()
        g["sympro_respondent"] = int(10_000_000 + i)
        chunks.append(g)
    return pd.concat(chunks, ignore_index=True)


def constraints_to_state_key_pairs(mdp: ClinicalMDP, constraint_set: set) -> set[tuple[str, int]]:
    """Map learned constraints from transient state ids to persistent state-key/action pairs."""
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


def _plot_preference_weights(results: dict, output_path: Path) -> None:
    weights = np.array(results["weights"], dtype=float)
    clusters = np.arange(weights.shape[0])
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(clusters - width / 2, weights[:, 0], width=width, label="QoL Weight", color="#1f77b4")
    ax.bar(clusters + width / 2, weights[:, 1], width=width, label="Survival Weight", color="#ff7f0e")

    ax.set_title("Learned Preference Weights by Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Weight Value")
    ax.set_xticks(clusters)
    ax.set_xticklabels([f"Cluster {i}" for i in clusters])
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_cluster_priors(results: dict, output_path: Path) -> None:
    priors = np.array(results["final_priors"], dtype=float)
    clusters = np.arange(len(priors))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(clusters, priors, color=["#2ca02c", "#d62728", "#9467bd", "#8c564b"][: len(priors)])

    ax.set_title("Inferred Cluster Prior Probabilities")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Prior Probability")
    ax.set_xticks(clusters)
    ax.set_xticklabels([f"Cluster {i}" for i in clusters])
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)

    for i, p in enumerate(priors):
        ax.text(i, p + 0.02, f"{p:.2f}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_action_feasibility(results: dict, output_path: Path) -> None:
    infeasible = pd.DataFrame(results.get("state_action_constraints", []))
    if infeasible.empty:
        action_labels = ACTION_NAMES[:4]
        feasibility = pd.Series({a: 1.0 for a in action_labels})
    else:
        action_labels = ACTION_NAMES[:4]
        counts = (
            infeasible[infeasible["eligibility_infeasible"]]
            .groupby("action")
            .size()
            .reindex(action_labels, fill_value=0)
        )
        total_states = max(1, int(infeasible["state_id"].nunique()))
        infeasible_rate = counts / total_states
        feasibility = 1.0 - np.clip(infeasible_rate, 0.0, 1.0)

    values = feasibility.values.reshape(1, -1)

    fig, ax = plt.subplots(figsize=(9, 2.8))
    img = ax.imshow(values, aspect="auto", cmap="YlGn", vmin=0.0, vmax=1.0)

    ax.set_title("Action Feasibility Rate Across Inferred States")
    ax.set_xlabel("Treatment Action")
    ax.set_yticks([0])
    ax.set_yticklabels(["Feasibility"])
    ax.set_xticks(np.arange(len(action_labels)))
    ax.set_xticklabels(action_labels)

    for j, val in enumerate(feasibility.values):
        ax.text(j, 0, f"{100.0 * val:.1f}%", ha="center", va="center", color="black", fontsize=10)

    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label("Feasibility Proportion")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_longitudinal_outcomes(df: pd.DataFrame, cluster_map: dict, output_path: Path, cluster_probabilities: list[dict] | None = None, num_clusters: int = 2) -> None:
    plot_df = df.copy()
    plot_df["alive_flag"] = (plot_df["survival_status"] == "Alive").astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    if cluster_probabilities:
        prob_df = pd.DataFrame(cluster_probabilities)
        if "sympro_respondent" in prob_df.columns:
            plot_df = plot_df.merge(prob_df, on="sympro_respondent", how="left")

    used_soft = False
    for cluster_id in range(num_clusters):
        prob_col = f"cluster_{cluster_id}_prob"
        if prob_col in plot_df.columns and plot_df[prob_col].notna().any():
            used_soft = True
            month_rows = []
            for month, g in plot_df.groupby("timepoint_month", sort=True):
                weights = np.clip(g[prob_col].to_numpy(dtype=float), 0.0, None)
                w_sum = float(weights.sum())
                if w_sum <= 0.0:
                    month_rows.append({"timepoint_month": int(month), "qol_mean": np.nan, "alive_rate": np.nan})
                    continue

                month_rows.append(
                    {
                        "timepoint_month": int(month),
                        "qol_mean": float(np.average(g["QoL"].to_numpy(dtype=float), weights=weights)),
                        "alive_rate": float(np.average(g["alive_flag"].to_numpy(dtype=float), weights=weights)),
                    }
                )
            qdf = pd.DataFrame(month_rows)

            axes[0].plot(
                qdf["timepoint_month"],
                qdf["qol_mean"],
                marker="o",
                linewidth=2,
                color=colors[cluster_id % len(colors)],
                label=f"Cluster {cluster_id}",
            )
            axes[1].plot(
                qdf["timepoint_month"],
                100.0 * qdf["alive_rate"],
                marker="s",
                linewidth=2,
                color=colors[cluster_id % len(colors)],
                label=f"Cluster {cluster_id}",
            )

    if not used_soft:
        plot_df["cluster"] = plot_df["sympro_respondent"].map(cluster_map)
        plot_df = plot_df.dropna(subset=["cluster"]).copy()
        plot_df["cluster"] = plot_df["cluster"].astype(int)

        month_qol = (
            plot_df.groupby(["cluster", "timepoint_month"], as_index=False)
            .agg(qol_mean=("QoL", "mean"))
            .sort_values(["cluster", "timepoint_month"])
        )
        month_alive = (
            plot_df.groupby(["cluster", "timepoint_month"], as_index=False)
            .agg(alive_rate=("alive_flag", "mean"))
            .sort_values(["cluster", "timepoint_month"])
        )

        for cluster_id in range(num_clusters):
            cdf_q = month_qol[month_qol["cluster"] == cluster_id]
            cdf_a = month_alive[month_alive["cluster"] == cluster_id]
            if cdf_q.empty or cdf_a.empty:
                continue
            axes[0].plot(
                cdf_q["timepoint_month"],
                cdf_q["qol_mean"],
                marker="o",
                linewidth=2,
                color=colors[cluster_id % len(colors)],
                label=f"Cluster {cluster_id}",
            )
            axes[1].plot(
                cdf_a["timepoint_month"],
                100.0 * cdf_a["alive_rate"],
                marker="s",
                linewidth=2,
                color=colors[cluster_id % len(colors)],
                label=f"Cluster {cluster_id}",
            )

    axes[0].set_title("Mean QoL Over Time")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Mean QoL")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].set_title("Alive Rate Over Time")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Alive Rate (%)")
    axes[1].set_ylim(0, 100)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.suptitle("Longitudinal Outcomes by Inferred Preference Cluster")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_true_false_positive_rates(results: dict, output_path: Path) -> None:
    metrics = results.get("constraint_detection", {})
    labels = ["True Positive Rate", "False Positive Rate"]
    values = [float(metrics.get("tpr", 0.0)), float(metrics.get("fpr", 0.0))]
    counts = [int(metrics.get("tp", 0)), int(metrics.get("fp", 0))]
    colors = ["#2ca02c", "#d62728"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors)

    ax.set_title("Learned Constraint True/False Positive Rates")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Rate")
    ymax = max(values) if values else 0.0
    top = min(1.0, ymax * 1.25 + 0.02) if ymax > 0 else 0.1
    ax.set_ylim(0.0, top)
    ax.grid(axis="y", alpha=0.25)

    for bar, v, c in zip(bars, values, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(v + top * 0.03, top * 0.98),
            f"{100.0 * v:.1f}% (n={c})",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_per_action_rates(results: dict, output_path: Path) -> None:
    metrics_df = pd.DataFrame(results.get("per_action_detection", []))
    if metrics_df.empty:
        return

    x = np.arange(len(metrics_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, metrics_df["tpr"], width=width, label="TPR", color="#2ca02c")
    ax.bar(x + width / 2, metrics_df["fpr"], width=width, label="FPR", color="#d62728")

    ax.set_title("Per-Treatment Constraint Detection Rates")
    ax.set_xlabel("Treatment Action")
    ax.set_ylabel("Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df["action"].tolist())
    yvals = np.concatenate([metrics_df["tpr"].to_numpy(dtype=float), metrics_df["fpr"].to_numpy(dtype=float)])
    ymax = float(np.nanmax(yvals)) if yvals.size > 0 else 0.0
    top = min(1.0, ymax * 1.25 + 0.02) if ymax > 0 else 0.1
    ax.set_ylim(0.0, top)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_threshold_sweep(results: dict, output_path: Path) -> None:
    sweep_df = pd.DataFrame(results.get("threshold_sweep", []))
    if sweep_df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(sweep_df["fpr"], sweep_df["tpr"], marker="o", color="#1f77b4", linewidth=2, label="Threshold Sweep")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Baseline")

    ax.set_title("Constraint Detection Threshold Sweep")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_restart_recall_precision(results: dict, output_path: Path) -> None:
    restart_df = pd.DataFrame(results.get("restart_metrics", []))
    if restart_df.empty:
        return

    restart_df = restart_df.sort_values("restart").copy()
    x = restart_df["restart"].to_numpy(dtype=int) + 1
    recall = restart_df["tpr"].to_numpy(dtype=float)
    precision = restart_df["precision"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, recall, marker="o", linewidth=2, color="#1f77b4", label="Recall (TPR)")
    ax.plot(x, precision, marker="s", linewidth=2, color="#ff7f0e", label="Precision")

    ax.set_title("Recall and Precision Across Restarts")
    ax.set_xlabel("Restart Index")
    ax.set_ylabel("Metric Value")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def generate_paper_figures(results: dict, df: pd.DataFrame, cluster_map: dict[int, int]) -> list[str]:
    RESULTS_DIR.mkdir(exist_ok=True)
    figure_paths = []

    pref_path = RESULTS_DIR / PLOT_FILENAMES["preferences"]
    _plot_preference_weights(results, pref_path)
    figure_paths.append(str(pref_path))

    prior_path = RESULTS_DIR / PLOT_FILENAMES["priors"]
    _plot_cluster_priors(results, prior_path)
    figure_paths.append(str(prior_path))

    feasibility_path = RESULTS_DIR / PLOT_FILENAMES["feasibility"]
    _plot_action_feasibility(results, feasibility_path)
    figure_paths.append(str(feasibility_path))

    longitudinal_path = RESULTS_DIR / PLOT_FILENAMES["longitudinal"]
    _plot_longitudinal_outcomes(
        df,
        cluster_map,
        longitudinal_path,
        cluster_probabilities=results.get("cluster_probabilities"),
        num_clusters=len(results.get("weights", [])) if results.get("weights") else 2,
    )
    figure_paths.append(str(longitudinal_path))

    tp_fp_path = RESULTS_DIR / PLOT_FILENAMES["tp_fp"]
    _plot_true_false_positive_rates(results, tp_fp_path)
    figure_paths.append(str(tp_fp_path))

    per_action_path = RESULTS_DIR / PLOT_FILENAMES["tp_fp_by_action"]
    _plot_per_action_rates(results, per_action_path)
    figure_paths.append(str(per_action_path))

    sweep_path = RESULTS_DIR / PLOT_FILENAMES["threshold_sweep"]
    _plot_threshold_sweep(results, sweep_path)
    figure_paths.append(str(sweep_path))

    rp_path = RESULTS_DIR / PLOT_FILENAMES["restart_recall_precision"]
    _plot_restart_recall_precision(results, rp_path)
    figure_paths.append(str(rp_path))

    return figure_paths


def run_lung_cancer_moci() -> dict:
    """Apply MOCI to the lung-cancer dataset and store the learned preference weights."""
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

    best = None
    restart_rows = []
    base_eval_rows = build_reference_eval_rows(df_train, state_to_idx_ref)
    inferred_votes = {(str(r["state_key"]), int(r["action_idx"])): 0 for r in base_eval_rows}
    ref_patient_ids = np.array(sorted(df_train["sympro_respondent"].unique()), dtype=int)

    for restart_idx in range(NUM_RESTARTS):
        seed = SEED_BASE + restart_idx
        rng = np.random.default_rng(seed)

        if BOOTSTRAP_RESTARTS:
            sampled_ids = rng.choice(ref_patient_ids, size=len(ref_patient_ids), replace=True)
            df_restart = bootstrap_patient_dataframe(df_train, sampled_ids)
        else:
            df_restart = df_train.copy()

        mdp, trajectories = build_clinical_mdp(df_restart, max_patients=None)
        np.random.seed(seed)
        inferred_constraints, final_weights, final_priors = moci.run_em_moci(
            mdp,
            trajectories,
            K=2,
            d_DKL=DKL_THRESHOLD,
            max_em_iters=EM_MAX_ITERS,
            candidate_subset_size=CANDIDATE_SUBSET_SIZE,
        )

        log_like = float(moci.calculate_joint_log_likelihood(mdp, trajectories, inferred_constraints, final_weights, final_priors))
        inferred_pairs = constraints_to_state_key_pairs(mdp, inferred_constraints)
        restart_prob = {
            (str(r["state_key"]), int(r["action_idx"])): 1.0 if (str(r["state_key"]), int(r["action_idx"])) in inferred_pairs else 0.0
            for r in base_eval_rows
        }
        eval_rows = build_eval_rows_from_probabilities(base_eval_rows, restart_prob, threshold=0.5)
        metrics = compute_tp_fp_metrics(eval_rows, set())
        restart_rows.append(
            {
                "restart": int(restart_idx),
                "seed": int(seed),
                "bootstrap": bool(BOOTSTRAP_RESTARTS),
                "joint_log_likelihood_avg": log_like,
                "n_constraints": int(len(inferred_constraints)),
                "tpr": float(metrics["tpr"]),
                "fpr": float(metrics["fpr"]),
                "precision": float(metrics["precision"]),
            }
        )

        for row in base_eval_rows:
            pair = (str(row["state_key"]), int(row["action_idx"]))
            if pair in inferred_pairs:
                inferred_votes[pair] += 1

        if (best is None) or (log_like > best["log_like"]):
            best = {
                "constraints": inferred_constraints,
                "weights": final_weights,
                "priors": final_priors,
                "log_like": log_like,
                "mdp": mdp,
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
            "train_max_patients": int(TRAIN_MAX_PATIENTS),
            "seed_base": int(SEED_BASE),
            "candidate_subset_size": None if CANDIDATE_SUBSET_SIZE is None else int(CANDIDATE_SUBSET_SIZE),
            "use_action_specific_thresholds": bool(USE_ACTION_SPECIFIC_THRESHOLDS),
        },
    }

    figure_paths = generate_paper_figures(results, df_train, cluster_map)
    results["generated_figures"] = figure_paths

    return results


def save_results(results: dict) -> None:
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
    for fig_path in results.get("generated_figures", []):
        print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    results = run_lung_cancer_moci()
    save_results(results)
    print("MOCI lung-cancer learning completed.")
    print(json.dumps({
        "constraint_type": results["constraint_type"],
        "inferred_constraints": results["inferred_constraints"],
        "state_action_summary": results["state_action_summary"][:10],
        "priors": results["final_priors"],
        "weights": results["weights"],
    }, indent=2))
