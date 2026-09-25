# MOCI: Multi-Objective Constraint Inference

MOCI is an EM-based inverse reinforcement learning algorithm. It takes a pool of
demonstrations from several experts whose preferences differ and are unknown, and
recovers two things together: (a) the reward weights of each expert cluster and
(b) one shared set of hidden hard constraints that every expert avoids. This
repository contains the core algorithm (`MOCI_IRL.py`) and applies it to two use
cases: a synthetic GridWorld and a lung-cancer treatment-planning MDP. It also
includes additional sensitivity, scalability and ablation experiments.

## Repository structure

```
MOCI_IRL.py                              Core MOCI algorithm 
requirements.txt                         Python dependencies
LICENSE                                  MIT license

use_case/
  GridWorld/
    gridworld_Env.py                     Multi-feature GridWorld MDP (sand/grass/rock/water)
                                         + plotting utilities
    Moci_GW.py                           MOCI applied to the GridWorld use case
  Lung_cancer/
    Moci_lung_cancer.py                  MOCI applied to the lung-cancer dataset
                                         (K=2 clusters, bootstrap-restart ensemble)

Additional_Experiments/
  Sensitivity_Scalability_analysis.py    FPR vs. dataset size, FPR vs. grid size,
                                         runtime vs. grid size across horizons
  Ablation_GW.py                         Oracle-reward, EM-iterations and K=1/2/3
                                         ablations on GridWorld

Results/                                 Saved outputs from previous runs (see "Outputs")
```

## Requirements

- Python 3.10+
- Dependencies in `requirements.txt`: `numpy`, `scipy`, `pandas`, `matplotlib`,
  `scikit-learn` (used by `Ablation_GW.py` for the adjusted Rand index) and `torch`.

```bash
python3 -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

## Running the experiments

Run every script from the repository root. Each script adds the root to `sys.path`
so it can import `MOCI_IRL`.

```bash
# GridWorld: an 8x8 grid with hidden WATER constraints and two experts
# (grass-preferring and rock-preferring), 20 demonstrations each. MOCI infers the
# shared constraint set and each expert's reward weights.
python3 use_case/GridWorld/Moci_GW.py

# Lung cancer: builds a clinical MDP from patient trajectories, runs MOCI with
# bootstrap-restart ensembling to learn state-action treatment constraints, compares
# them with clinical eligibility, then renders the paper figures.
python3 use_case/Lung_cancer/Moci_lung_cancer.py


# Sensitivity / scalability sweeps
python3 Additional_Experiments/Sensitivity_Scalability_analysis.py

# GridWorld ablations (oracle reward, EM-iteration budget, K-cluster capacity)
python3 Additional_Experiments/Ablation_GW.py
```


## Outputs

Every script writes into `Results/`. 


## License

MIT, see `LICENSE`.
