import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent
INPUT_PATH = BASE / "lung_cancer_cirl_dataset.csv"
OUTPUT_PATH = BASE / "lung_cancer_cirl_dataset_realistic_nice.csv"

if not INPUT_PATH.exists():
    INPUT_PATH = BASE / "lung_cancer_data.csv"
if not INPUT_PATH.exists():
    INPUT_PATH = BASE / "lung_cancer_cirl_dataset_realistic.csv"


def perturb_profile(row: pd.Series) -> pd.Series:
    out = row.copy()

    age = float(out.get("Leeftijd_bj_diagnose", 65.0) or 65.0)
    out["Leeftijd_bj_diagnose"] = float(np.clip(age + np.random.normal(0.0, 4.0), 35.0, 90.0))

    perf = int(out.get("perf_status", 0) or 0)
    if np.random.rand() < 0.25:
        perf = perf + int(np.random.choice([-1, 1]))
    out["perf_status"] = int(np.clip(perf, 0, 3))

    tox = int(out.get("ctcae_toxicity_grade", 0) or 0)
    if np.random.rand() < 0.22:
        tox = tox + int(np.random.choice([0, 1]))
    out["ctcae_toxicity_grade"] = int(np.clip(tox, 0, 4))

    pdl1 = float(out.get("pdl1_expression_pct", 0.0) or 0.0)
    out["pdl1_expression_pct"] = float(np.clip(pdl1 + np.random.normal(0.0, 8.0), 0.0, 100.0))

    egfr_lab = float(out.get("egfr_lab_ml_min", 80.0) or 80.0)
    out["egfr_lab_ml_min"] = float(np.clip(egfr_lab + np.random.normal(0.0, 10.0), 20.0, 130.0))

    fev1 = float(out.get("fev1_pct_predicted", 80.0) or 80.0)
    out["fev1_pct_predicted"] = float(np.clip(fev1 + np.random.normal(0.0, 9.0), 20.0, 110.0))

    if np.random.rand() < 0.06:
        out["egfr_mutation"] = int(1 - int(out.get("egfr_mutation", 0) or 0))
    if np.random.rand() < 0.05:
        out["alk_translocation"] = int(1 - int(out.get("alk_translocation", 0) or 0))

    # Keep enough mutation-positive profiles so Targeted trajectories are identifiable.
    if int(out.get("egfr_mutation", 0) or 0) == 0 and int(out.get("alk_translocation", 0) or 0) == 0:
        if np.random.rand() < 0.16:
            out["egfr_mutation"] = 1

    return out


def sample_action_from_eligibility(row: pd.Series, allow_surgery: bool = True) -> str:
    m_targeted, m_immuno, m_chemo, m_surgery = compute_action_eligibility(row)
    eligible = {
        "Targeted": m_targeted == 1,
        "Immuno": m_immuno == 1,
        "Chemo": m_chemo == 1,
        "Surgery": (m_surgery == 1) and allow_surgery,
    }

    has_mut = int(row.get("egfr_mutation", 0) or 0) == 1 or int(row.get("alk_translocation", 0) or 0) == 1
    pdl1 = float(row.get("pdl1_expression_pct", 0.0) or 0.0)
    stage = str(row.get("tnm", "T1")).upper()
    meta = str(row.get("meta", "No")).lower()

    weights = {
        "Targeted": 0.32,
        "Immuno": 0.24,
        "Chemo": 0.28,
        "Surgery": 0.16,
    }
    if has_mut:
        weights["Targeted"] += 0.38
        weights["Chemo"] -= 0.10
        weights["Immuno"] -= 0.03
    if pdl1 >= 10:
        weights["Immuno"] += 0.20
        weights["Chemo"] -= 0.06
    if stage in {"T1", "T2"} and meta != "yes":
        weights["Surgery"] += 0.18

    actions = []
    probs = []
    for action in ["Targeted", "Immuno", "Chemo", "Surgery"]:
        if eligible[action]:
            actions.append(action)
            probs.append(max(0.01, weights[action]))

    if not actions:
        return "Chemo"

    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return str(np.random.choice(actions, p=probs))


def assign_therapy_from_profile(row: pd.Series) -> str:
    return sample_action_from_eligibility(row, allow_surgery=True)


def next_therapy(current: str, row: pd.Series) -> str:
    continue_prob = {
        "Surgery": 0.85,
        "Targeted": 0.60,
        "Immuno": 0.58,
        "Chemo": 0.48,
    }.get(current, 0.55)

    if np.random.rand() < continue_prob:
        return current

    candidate = sample_action_from_eligibility(row, allow_surgery=False)
    if candidate == current and np.random.rand() < 0.6:
        if current == "Chemo":
            has_mut = int(row.get("egfr_mutation", 0) or 0) == 1 or int(row.get("alk_translocation", 0) or 0) == 1
            return "Targeted" if has_mut else "Immuno"
        return "Chemo"
    return candidate


def compute_action_eligibility(row: pd.Series) -> tuple[int, int, int, int]:
    """Compute eligibility masks from clinical profile, independent of chosen therapy."""
    stage = str(row.get("tnm", "T1")).upper()
    perf = int(row.get("perf_status", 0) or 0)
    age = float(row.get("Leeftijd_bj_diagnose", 65) or 65)
    meta = str(row.get("meta", "No")).lower()
    tox = int(row.get("ctcae_toxicity_grade", 0) or 0)
    egfr_mut = int(row.get("egfr_mutation", 0) or 0)
    alk = int(row.get("alk_translocation", 0) or 0)
    pdl1 = float(row.get("pdl1_expression_pct", 0.0) or 0.0)
    egfr_lab = float(row.get("egfr_lab_ml_min", 80.0) or 80.0)
    fev1 = float(row.get("fev1_pct_predicted", 80.0) or 80.0)

    action_mask_surgery = int(stage in {"T1", "T2"} and perf <= 1 and age <= 75 and meta != "yes" and tox < 3)
    action_mask_targeted = int((egfr_mut == 1 or alk == 1) and perf <= 2 and tox < 3)
    action_mask_immuno = int(pdl1 >= 1 and perf <= 2 and tox < 3)
    action_mask_chemo = int(egfr_lab >= 50 and fev1 >= 45 and perf <= 2 and tox < 3)

    return action_mask_targeted, action_mask_immuno, action_mask_chemo, action_mask_surgery


def compute_death_month(therapy: str, row: pd.Series) -> int | None:
    stage = str(row.get("tnm", "T1")).upper()
    perf = int(row.get("perf_status", 0) or 0)
    size = float(row.get("tumor_size", 0.0) or 0.0)
    meta = str(row.get("meta", "No")).lower()
    age = float(row.get("Leeftijd_bj_diagnose", 65) or 65)
    tox = int(row.get("ctcae_toxicity_grade", 0) or 0)
    egfr_mut = int(row.get("egfr_mutation", 0) or 0)
    alk = int(row.get("alk_translocation", 0) or 0)
    pdl1 = float(row.get("pdl1_expression_pct", 0.0) or 0.0)
    egfr_lab = float(row.get("egfr_lab_ml_min", 0.0) or 0.0)
    fev1 = float(row.get("fev1_pct_predicted", 0.0) or 0.0)

    base_risk = {"Surgery": 0.12, "Targeted": 0.18, "Immuno": 0.20, "Chemo": 0.28}[therapy]
    risk = base_risk
    if stage in {"T3", "T4"}:
        risk += 0.15
    if meta == "yes":
        risk += 0.18
    if perf > 1:
        risk += 0.12
    if age > 70:
        risk += 0.08
    if tox >= 3:
        risk += 0.15
    if therapy == "Surgery" and stage in {"T1", "T2"} and perf <= 1 and meta != "yes":
        risk -= 0.10
    if therapy == "Targeted" and (egfr_mut == 1 or alk == 1):
        risk -= 0.08
    if therapy == "Immuno" and pdl1 >= 10:
        risk -= 0.06
    if therapy == "Chemo" and (egfr_lab < 50 or fev1 < 45 or perf > 2):
        risk += 0.12

    risk = float(np.clip(risk, 0.02, 0.75))
    r = np.random.random()
    if r < risk * 0.20:
        return 12
    if r < risk * 0.38:
        return 18
    if r < risk * 0.58:
        return 24
    return None


def qol_curve(therapy: str, month: int, dead: bool) -> float:
    if dead:
        return 0.0
    if therapy == "Surgery":
        if month == 0:
            return float(np.random.uniform(20, 35))
        if month == 6:
            return float(np.random.uniform(24, 40))
        if month == 12:
            return float(np.random.uniform(60, 80))
        if month == 18:
            return float(np.random.uniform(70, 90))
        return float(np.random.uniform(75, 95))
    if therapy == "Targeted":
        if month == 0:
            return float(np.random.uniform(55, 75))
        if month == 6:
            return float(np.random.uniform(60, 80))
        if month == 12:
            return float(np.random.uniform(65, 85))
        if month == 18:
            return float(np.random.uniform(70, 88))
        return float(np.random.uniform(68, 90))
    if therapy == "Immuno":
        if month == 0:
            return float(np.random.uniform(50, 70))
        if month == 6:
            return float(np.random.uniform(58, 76))
        if month == 12:
            return float(np.random.uniform(62, 82))
        if month == 18:
            return float(np.random.uniform(60, 82))
        return float(np.random.uniform(58, 88))
    if month == 0:
        return float(np.random.uniform(40, 70))
    if month == 6:
        return float(np.random.uniform(35, 65))
    if month == 12:
        return float(np.random.uniform(30, 60))
    if month == 18:
        return float(np.random.uniform(25, 55))
    return float(np.random.uniform(20, 50))


def assign_response(therapy: str, month: int, dead: bool) -> str:
    if dead:
        return "Progression"
    if therapy == "Surgery":
        if month in {0, 6}:
            return "Partial response"
        if month in {12, 18, 24}:
            return np.random.choice(["Disease control", "Stable disease", "Partial response"], p=[0.45, 0.35, 0.20])
    if therapy == "Targeted":
        if month in {0, 6}:
            return np.random.choice(["Partial response", "Disease control", "Stable disease"], p=[0.55, 0.25, 0.20])
        return np.random.choice(["Stable disease", "Disease control", "Progression"], p=[0.45, 0.35, 0.20])
    if therapy == "Immuno":
        if month in {0, 6}:
            return np.random.choice(["Disease control", "Stable disease", "Partial response"], p=[0.40, 0.35, 0.25])
        return np.random.choice(["Stable disease", "Disease control", "Progression"], p=[0.40, 0.35, 0.25])
    if month in {0, 6}:
        return np.random.choice(["Partial response", "Stable disease", "Progression"], p=[0.30, 0.40, 0.30])
    return np.random.choice(["Stable disease", "Progression", "Disease control"], p=[0.35, 0.40, 0.25])


def build_dataset() -> pd.DataFrame:
    base = pd.read_csv(INPUT_PATH)
    patients = base.sample(n=1000, replace=True, random_state=42).reset_index(drop=True)
    rows = []
    for pid, row in patients.iterrows():
        row = perturb_profile(row)
        patient_id = 1000 + pid + 1
        initial_therapy = assign_therapy_from_profile(row)
        death_month = compute_death_month(initial_therapy, row)
        current_therapy = initial_therapy
        switch_prob = np.random.rand()

        for month in [0, 6, 12, 18, 24]:
            dead = death_month is not None and month >= death_month
            if month == 0:
                therapy = initial_therapy
            elif month == 6 and current_therapy in {"Targeted", "Immuno", "Chemo"} and switch_prob < 0.55 and not dead:
                therapy = next_therapy(current_therapy, row)
            elif month == 12 and current_therapy in {"Targeted", "Immuno", "Chemo"} and not dead:
                therapy = next_therapy(current_therapy, row) if np.random.rand() < 0.50 else current_therapy
            elif month == 18 and current_therapy in {"Targeted", "Immuno", "Chemo"} and not dead:
                therapy = next_therapy(current_therapy, row) if np.random.rand() < 0.45 else current_therapy
            else:
                therapy = current_therapy
            current_therapy = therapy

            treatment_intensity = {"Surgery": 1, "Targeted": 2, "Immuno": 2, "Chemo": 3}.get(therapy, 1)
            action_mask_targeted, action_mask_immuno, action_mask_chemo, action_mask_surgery = compute_action_eligibility(row)

            q = qol_curve(therapy, month, dead)
            if month in {12, 18, 24} and death_month is None:
                q = max(q, 50.0)

            survival_status = "Dead" if dead else "Alive"
            event_indicator = 1 if dead else 0
            treatment_response = assign_response(therapy, month, dead)
            s_time = death_month if death_month is not None and month >= death_month else 24 if not dead else 0
            if s_time == 0:
                s_time = 24

            row_dict = {
                "sympro_respondent": patient_id,
                "timepoint_month": month,
                "bas_dem_gender": row.get("bas_dem_gender", "Female"),
                "Leeftijd_bj_diagnose": float(row.get("Leeftijd_bj_diagnose", 65.0) or 65.0),
                "Roken": row.get("Roken", "Never"),
                "tnm": row.get("tnm", "T1"),
                "perf_status": int(row.get("perf_status", 0) or 0),
                "egfr_mutation": int(row.get("egfr_mutation", 0) or 0),
                "alk_translocation": int(row.get("alk_translocation", 0) or 0),
                "pdl1_expression_pct": float(row.get("pdl1_expression_pct", 0.0) or 0.0),
                "egfr_lab_ml_min": float(row.get("egfr_lab_ml_min", 80.0) or 80.0),
                "fev1_pct_predicted": float(row.get("fev1_pct_predicted", 80.0) or 80.0),
                "ctcae_toxicity_grade": int(row.get("ctcae_toxicity_grade", 0) or 0),
                "action_mask_targeted": action_mask_targeted,
                "action_mask_immuno": action_mask_immuno,
                "action_mask_chemo": action_mask_chemo,
                "action_mask_surgery": action_mask_surgery,
                "therapy": therapy,
                "treatment_intensity": treatment_intensity,
                "tumor_size": float(row.get("tumor_size", 30.0) or 30.0) + np.random.uniform(-5, 5),
                "meta": row.get("meta", "No"),
                "tumor_size_prev": float(row.get("tumor_size_prev", 30.0) or 30.0),
                "perf_prev": float(row.get("perf_prev", 0.0) or 0.0),
                "QoL": round(float(q), 2),
                "survival_time_months": int(s_time),
                "survival_status": survival_status,
                "event_indicator": event_indicator,
                "treatment_response": treatment_response,
                "treatment_line": 1 if month <= 6 else 2 if month <= 12 else 3 if month <= 18 else 4,
                "treatment_ongoing": int(1 if not dead else 0),
                "eligibility_surgery": int(1 if action_mask_surgery else 0),
                "eligibility_targeted": int(1 if action_mask_targeted else 0),
                "eligibility_immuno": int(1 if action_mask_immuno else 0),
                "eligibility_chemo": int(1 if action_mask_chemo else 0),
                "trajectory_clinical_label": "NICE_surgery_candidate" if therapy == "Surgery" else "NICE_targeted_therapy_candidate" if therapy == "Targeted" else "NICE_immunotherapy_candidate" if therapy == "Immuno" else "NICE_chemo_candidate",
            }
            rows.append(row_dict)

    out = pd.DataFrame(rows)
    out = out.sort_values(["sympro_respondent", "timepoint_month"]).reset_index(drop=True)
    return out


def main():
    df = build_dataset()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved NICE-informed realistic dataset to: {OUTPUT_PATH}")
    print("Rows:", len(df), "Patients:", df["sympro_respondent"].nunique())
    print("Therapies:", df["therapy"].value_counts().to_dict())
    alive_12 = df[df["timepoint_month"] == 12]["survival_status"].eq("Alive").mean()
    alive_18 = df[df["timepoint_month"] == 18]["survival_status"].eq("Alive").mean()
    print("Alive at 12 months:", round(alive_12 * 100, 2), "%")
    print("Alive at 18 months:", round(alive_18 * 100, 2), "%")


if __name__ == "__main__":
    main()
