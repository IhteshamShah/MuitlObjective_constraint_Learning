"""Generates the publication figures for the MOCI-IRL lung-cancer use case.

Loads the summary JSON and result CSVs produced by moci_lung_cancer.py (and optional
ablation/diagnostic runs), then renders and saves ten figures as PDF+PNG pairs: learned
reward weights per cluster, longitudinal quality-of-life and survival outcomes, constraint
detection performance (confusion matrix, per-action rates, ROC), recall by clinical reason,
lambda-sweep and restart-stability sensitivity analyses, an ablation of low-TPR remedies,
and diagnostics explaining low recall. Can be run standalone or invoked from the main
experiment script.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = RESULTS / "paper_figures"
PATH_DIAG = HERE / "results_diagnostics_path_100"
DATA_CSV = HERE / "lung_cancer_cirl_dataset_realistic_nice.csv"
SAVED: list[str] = []


def configure(results_dir=None):
    """Args: results_dir. Returns: None; sets module-level RESULTS/OUT and clears SAVED."""
    global RESULTS, OUT
    RESULTS = Path(results_dir).resolve() if results_dir is not None else HERE / "results"
    OUT = RESULTS / "paper_figures"
    OUT.mkdir(parents=True, exist_ok=True)
    SAVED.clear()

SURFACE = "#ffffff"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
VIOLET, AQUA = "#4a3aa7", "#1baf7a"
GRAY = "#b9b8b0"
CLUSTER_COLORS = {0: VIOLET, 1: AQUA}
BLUE_RAMP = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#5598e7", "#256abf", "#104281"])

rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9.5,
        "axes.linewidth": 0.8,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def wilson_ci(k, n, z=1.96):
    """Args: k, n, z. Returns: (lower, upper) 95% Wilson confidence bounds."""
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def style(ax, grid="y"):
    """Args: ax, grid. Returns: None; applies shared axis styling to ax."""
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_axisbelow(True)
    if grid == "y":
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    elif grid == "x":
        ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    elif grid == "both":
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)


def title(ax, letter, text, fontsize=None, pad=10):
    """Args: ax, letter, text, fontsize, pad. Returns: None; sets a left-aligned panel title on ax."""
    kw = {} if fontsize is None else {"fontsize": fontsize}
    ax.set_title(f"({letter}) {text}", loc="left", pad=pad, **kw)


def footnote(fig, text):
    """Args: fig, text. Returns: None; adds a footnote to fig."""
    fig.text(0.995, 0.005, text, ha="right", va="bottom", fontsize=7.5, color=MUTED)


def save(fig, name):
    """Args: fig, name. Returns: None; saves fig as name.pdf and name.png and records the paths in SAVED."""
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    SAVED.extend([str(OUT / f"{name}.pdf"), str(OUT / f"{name}.png")])
    print("Saved figure:", OUT / f"{name}.png")


def pct(x, digits=1):
    """Args: x, digits. Returns: x formatted as a percentage string."""
    return f"{100 * x:.{digits}f}%"


def bar_label(ax, x, y, text, dy=0.02, bold=False, fontsize=8.5, color=INK2, **kw):
    """Args: ax, x, y, text, dy, bold, fontsize, color, **kw. Returns: None; draws a text label above a bar on ax."""
    ax.text(x, y + dy, text, ha="center", va="bottom", fontsize=fontsize, color=color, linespacing=1.25,
            fontweight="bold" if bold else "normal", **kw)


def load():
    """Args: none. Returns: (summary, tables) loaded from RESULTS (and PATH_DIAG) as a dict and a dict of DataFrames."""
    summary = json.loads((RESULTS / "summary.json").read_text())
    tables = {
        "restart": pd.read_csv(RESULTS / "constraint_restart_metrics.csv"),
        "per_action": pd.read_csv(RESULTS / "constraint_detection_per_action.csv"),
        "detect": pd.read_csv(RESULTS / "constraint_detection_metrics.csv").iloc[0],
        "sweep_thr": pd.read_csv(RESULTS / "constraint_threshold_sweep.csv"),
        "lam": pd.read_csv(RESULTS / "lambda_sweep.csv") if (RESULTS / "lambda_sweep.csv").exists() else None,
        "diag": pd.read_csv(RESULTS / "diagnostics_low_tpr.csv") if (RESULTS / "diagnostics_low_tpr.csv").exists() else None,
        "pairs": pd.read_csv(RESULTS / "eligibility_vs_learned_constraints.csv"),
        "assign": pd.read_csv(RESULTS / "cluster_assignments.csv"),
        "ablation": pd.read_csv(RESULTS / "ablation_low_tpr.csv") if (RESULTS / "ablation_low_tpr.csv").exists() else None,
        "diag_path": pd.read_csv(PATH_DIAG / "diagnostics_low_tpr.csv") if (PATH_DIAG / "diagnostics_low_tpr.csv").exists() else None,
    }
    return summary, tables


def fig_preferences(summary, t):
    """Args: summary, t. Saves fig1_preference_model.pdf/.png."""
    from matplotlib.ticker import MaxNLocator

    w = np.array(summary["weights"], dtype=float)
    features = ["Quality of life", "Survival"]
    feature_colors = [BLUE, ORANGE]
    n_clusters = w.shape[0]
    span = max(float(np.abs(w).max()), 1e-3)

    fig, a = plt.subplots(figsize=(4.8, 3.7))

    step = 2.7
    for f, color in enumerate(feature_colors):
        x = np.arange(n_clusters) * step + f
        a.bar(x, w[:, f], width=0.62, color=color, edgecolor=SURFACE, linewidth=0.8, zorder=3)
        for xi, v in zip(x, w[:, f]):
            bar_label(a, xi, max(v, 0), f"{v:.2f}", dy=0.02 * span)

    a.axhline(0, color=AXIS, linewidth=0.8, zorder=2)
    a.set_xticks([])
    a.set_xlim(-0.75, (n_clusters - 1) * step + 1.75)
    a.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, min_n_ticks=3, steps=[1, 2, 5, 10]))
    a.set_ylim(min(0, w.min() * 1.2), max(0, w.max()) * 1.32)
    a.set_ylabel("Learned reward weight")
    for k in range(n_clusters):
        xs = (k * step, k * step + 1)
        a.plot(xs, [-0.04, -0.04], color=AXIS, linewidth=0.8, clip_on=False, transform=a.get_xaxis_transform())
        a.text(np.mean(xs), -0.08, f"Cluster {k}", ha="center", va="top", fontsize=8.5, color=INK2,
               transform=a.get_xaxis_transform())
    a.legend(handles=[Patch(color=c, label=n) for n, c in zip(features, feature_colors)],
             loc="upper left", ncol=2, handlelength=1.0, handleheight=1.0, bbox_to_anchor=(0.0, 1.02))
    style(a)
    a.set_title("Reward weights per cluster", loc="left", pad=10)

    fig.tight_layout()
    save(fig, "fig1_preference_model")


def weighted_curve(g, w, col):
    """Args: g, w, col. Returns: (mean, n_eff, x, sum_weights) for column col of g weighted by w."""
    w = np.clip(w, 0, None)
    sw = w.sum()
    if sw <= 0:
        return np.nan, np.nan, np.nan, 0.0
    n_eff = sw**2 / np.sum(w**2)
    x = g[col].to_numpy(float)
    m = float(np.sum(w * x) / sw)
    return m, n_eff, x, sw


def fig_longitudinal(summary, t):
    """Args: summary, t. Saves fig2_longitudinal_outcomes.pdf/.png."""
    n_train = int(summary["n_patients_used_for_training"])
    df = pd.read_csv(DATA_CSV)
    df = df[df["QoL"].notna()].sort_values(["sympro_respondent", "timepoint_month"])
    ids = sorted(df["sympro_respondent"].unique())[:n_train]
    df = df[df["sympro_respondent"].isin(ids)].copy()
    df["alive"] = (df["survival_status"] == "Alive").astype(float)
    probs = pd.DataFrame(summary["cluster_probabilities"])
    df = df.merge(probs, on="sympro_respondent", how="left")
    priors = summary["final_priors"]
    shown = [k for k, p in enumerate(priors) if p > 0.01]

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    months = sorted(df["timepoint_month"].unique())
    for k in shown:
        rows = {"qol": [], "alive": []}
        for mth in months:
            g = df[df["timepoint_month"] == mth]
            w = np.clip(g[f"cluster_{k}_prob"].to_numpy(float), 0, None)
            sw = w.sum()
            n_eff = sw**2 / np.sum(w**2)
            q = g["QoL"].to_numpy(float)
            m = np.sum(w * q) / sw
            sd = np.sqrt(np.sum(w * (q - m) ** 2) / sw)
            half = 1.96 * sd / np.sqrt(n_eff)
            rows["qol"].append((m, m - half, m + half))
            p_alive = np.sum(w * g["alive"].to_numpy(float)) / sw
            lo, hi = wilson_ci(p_alive * n_eff, n_eff)
            rows["alive"].append((100 * p_alive, 100 * lo, 100 * hi))
        for ax, key in zip(axes, ("qol", "alive")):
            arr = np.array(rows[key])
            ax.fill_between(months, arr[:, 1], arr[:, 2], color=CLUSTER_COLORS[k], alpha=0.12, linewidth=0, zorder=2)
            ax.plot(months, arr[:, 0], color=CLUSTER_COLORS[k], linewidth=2, solid_capstyle="round", zorder=3)
            ax.plot(months, arr[:, 0], "o", color=CLUSTER_COLORS[k], markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
            ax.text(months[-1], arr[-1, 0], f"  Cluster {k}", va="center", ha="left", fontsize=8.5, color=INK2)

    axes[0].set_ylabel("Mean QoL score")
    axes[1].set_ylabel("Alive (%)")
    axes[1].set_ylim(0, 105)
    for ax, letter, text in zip(axes, "ab", ("Quality of life over time", "Survival over time")):
        ax.set_xlabel("Months since baseline")
        ax.set_xticks(months)
        ax.set_xlim(months[0] - 0.5, months[-1] + max(3, 0.25 * months[-1]))
        style(ax)
        title(ax, letter, text)
    omitted = [k for k in range(len(priors)) if k not in shown]
    note = "Shaded band: 95% interval, patients weighted by cluster responsibility."
    if omitted:
        note += " " + ", ".join(f"Cluster {k}" for k in omitted) + f" holds no patients (pi < 1e-3) and is omitted."
    footnote(fig, note)
    fig.tight_layout(rect=(0, 0.03, 1, 1), w_pad=2)
    save(fig, "fig2_longitudinal_outcomes")


def fig_detection(summary, t):
    """Args: summary, t. Saves fig3_detection_rates.pdf/.png."""
    d = t["detect"]
    tp, fp, tn, fn = (int(d[k]) for k in ("tp", "fp", "tn", "fn"))
    P, N = tp + fn, fp + tn

    fig, (a, b) = plt.subplots(1, 2, figsize=(14 / 2.54, 3.5 / 2.54), gridspec_kw={"width_ratios": [1, 1.6]})

    counts = np.array([[tp, fn], [fp, tn]])
    share = counts / counts.sum(axis=1, keepdims=True)
    a.pcolormesh(share[::-1], cmap=BLUE_RAMP, vmin=0, vmax=1, edgecolors=SURFACE, linewidth=1.4)
    for i in range(2):
        for j in range(2):
            s = share[i, j]
            a.text(j + 0.5, 1.5 - i, f"{counts[i, j]:,}\n{pct(s)}", ha="center", va="center", fontsize=5.8,
                   fontweight="bold", color="white" if s > 0.5 else INK, linespacing=1.15)
    a.set_xticks([0.5, 1.5])
    a.set_xticklabels(["Flagged", "Not flagged"], fontsize=5.6, fontweight="bold", color=INK)
    a.set_yticks([0.5, 1.5])
    a.set_yticklabels([f"Feasible (n={N:,})", f"Infeasible (n={P:,})"], fontsize=5.6, fontweight="bold", color=INK)
    a.set_xlabel("Learned constraint", labelpad=2, fontsize=5.9, fontweight="bold", color=INK)
    a.set_ylabel("Ground truth", fontsize=5.9, fontweight="bold", labelpad=2, color=INK)
    a.tick_params(length=0, pad=2)
    for s in a.spines.values():
        s.set_visible(False)
    title(a, "a", "Confusion matrix", fontsize=7.2, pad=4)

    items = [
        ("TPR", tp, P, BLUE, "sensitivity"),
        ("FNR", fn, P, ORANGE, "miss rate"),
        ("TNR", tn, N, BLUE, "specificity"),
        ("FPR", fp, N, ORANGE, "false alarm"),
    ]
    x = np.array([0, 1, 3.15, 4.15])
    rates = np.array([k / n for _, k, n, _, _ in items])
    cis = [wilson_ci(k, n) for _, k, n, _, _ in items]
    lo = np.clip(rates - [c[0] for c in cis], 0, None)
    hi = np.clip([c[1] for c in cis] - rates, 0, None)
    b.bar(x, rates, width=0.62, color=[c for *_, c, _ in items], edgecolor=SURFACE, linewidth=0.6, zorder=3)
    b.errorbar(x, rates, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=0.7, capsize=1.6, capthick=0.7, zorder=4)
    for xi, (name, k, n, _, _), r, h in zip(x, items, rates, hi):
        bar_label(b, xi, r + h, f"{pct(r)} (n={k:,})", dy=0.03, fontsize=5.0, bold=True, color=INK)
    b.set_xticks(x)
    b.set_xticklabels([f"{name}\n{sub}" for name, _, _, _, sub in items], fontsize=5.0, fontweight="bold",
                      linespacing=1.35, color=INK)
    b.tick_params(axis="x", pad=3, length=2)
    b.tick_params(axis="y", pad=1.5, length=2)
    b.set_ylim(0, 1.42)
    b.set_yticks(np.arange(0, 1.01, 0.2))
    b.set_yticklabels([f"{v:.1f}" for v in np.arange(0, 1.01, 0.2)], fontsize=5.2, fontweight="bold", color=INK)
    b.set_ylabel("Rate", fontsize=5.9, fontweight="bold", labelpad=1, color=INK)
    leg = b.legend(handles=[Patch(color=BLUE, label="Correct decision"), Patch(color=ORANGE, label="Error")],
                   loc="upper left", ncol=2, handlelength=0.8, handleheight=0.8, handletextpad=0.3,
                   columnspacing=0.8, bbox_to_anchor=(0.0, 1.06), fontsize=5.6, borderaxespad=0.1)
    for txt in leg.get_texts():
        txt.set_fontweight("bold")
        txt.set_color(INK)
    style(b)
    title(b, "b", "Detection rates", fontsize=7.2, pad=4)

    fig.tight_layout(w_pad=0.8)
    save(fig, "fig3_detection_rates")


def fig_per_action(summary, t):
    """Args: summary, t. Saves fig4_per_action_detection.pdf/.png."""
    pa = t["per_action"]
    x = np.arange(len(pa))
    fig, (a, b) = plt.subplots(1, 2, figsize=(14 / 2.54, 3.5 / 2.54), gridspec_kw={"width_ratios": [1.6, 1]})

    off, bw = 0.2, 0.34
    for j, (metric, k_col, n_fn, color) in enumerate(
        (("TPR", "tp", lambda r: r.tp + r.fn, BLUE), ("FPR", "fp", lambda r: r.fp + r.tn, ORANGE))
    ):
        vals, los, his, labels = [], [], [], []
        for r in pa.itertuples():
            k, n = getattr(r, k_col), n_fn(r)
            lo_, hi_ = wilson_ci(k, n)
            v = k / n
            vals.append(v); los.append(max(0.0, v - lo_)); his.append(max(0.0, hi_ - v))
        vals = np.array(vals)
        xs = x + (j - 0.5) * 2 * off
        a.bar(xs, vals, width=bw, color=color, edgecolor=SURFACE, linewidth=0.6, zorder=3)
        a.errorbar(xs, vals, yerr=[los, his], fmt="none", ecolor=INK2, elinewidth=0.6, capsize=1.4, capthick=0.6, zorder=4)
        for xi, v, h in zip(xs, vals, his):
            bar_label(a, xi, v + h, pct(v, 0) if v >= 0.005 else "0%", dy=0.03, fontsize=5.0, bold=True, color=INK)
    a.set_xticks(x)
    a.set_xticklabels(pa["action"], fontsize=5.6, fontweight="bold", color=INK)
    a.tick_params(axis="both", length=2, pad=1.5)
    a.set_ylim(0, 1.22)
    a.set_yticks(np.arange(0, 1.01, 0.2))
    a.set_yticklabels([f"{v:.1f}" for v in np.arange(0, 1.01, 0.2)], fontsize=5.2, fontweight="bold", color=INK)
    a.set_ylabel("Rate", fontsize=5.9, fontweight="bold", labelpad=1, color=INK)
    leg = a.legend(handles=[Patch(color=BLUE, label="True positive rate"), Patch(color=ORANGE, label="False positive rate")],
                   loc="upper center", ncol=2, handlelength=0.8, handleheight=0.8, handletextpad=0.3,
                   columnspacing=0.8, bbox_to_anchor=(0.5, 1.0), borderaxespad=0.2, fontsize=5.2)
    for txt in leg.get_texts():
        txt.set_fontweight("bold")
        txt.set_color(INK)
    style(a)
    title(a, "a", "Detection by treatment", fontsize=7.2, pad=4)

    prev = (pa["tp"] + pa["fn"]) / pa["n_pairs"]
    b.bar(x, prev, width=0.5, color=GRAY, edgecolor=SURFACE, linewidth=0.6, zorder=3)
    for xi, v, r in zip(x, prev, pa.itertuples()):
        bar_label(b, xi, v, f"{pct(v, 0)} ({r.tp + r.fn})", dy=0.03, fontsize=5.0, bold=True, color=INK)
    b.set_xticks(x)
    b.set_xticklabels(pa["action"], rotation=25, ha="right", fontsize=5.6, fontweight="bold", color=INK)
    b.tick_params(axis="both", length=2, pad=1.5)
    b.set_ylim(0, 1.22)
    b.set_yticks(np.arange(0, 1.01, 0.2))
    b.set_yticklabels([f"{v:.1f}" for v in np.arange(0, 1.01, 0.2)], fontsize=5.2, fontweight="bold", color=INK)
    b.set_ylabel("Share of pairs", fontsize=5.9, fontweight="bold", labelpad=1, color=INK)
    style(b)
    title(b, "b", "Clinically infeasible", fontsize=7.2, pad=4)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig4_per_action_detection")


def fig_reasons(summary, t):
    """Args: summary, t. Saves fig5_recall_by_clinical_reason.pdf/.png."""
    pairs = t["pairs"]
    gt = pairs[pairs["eligibility_infeasible"]]
    g = gt.groupby(["action", "clinical_reason"]).agg(n=("inferred_infeasible", "size"), det=("inferred_infeasible", "sum")).reset_index()
    g["missed"] = g["n"] - g["det"]
    pretty = {"allowed": "not explained by rule set", "toxicity_grade>=3": "toxicity grade ≥ 3", "ECOG/perf_status > 2": "ECOG > 2",
              "EGFR/ALK not eligible": "no EGFR/ALK driver", "PD-L1 < 1%": "PD-L1 < 1%", "not first-line month": "not first-line month"}
    g["label"] = g["action"] + " · " + g["clinical_reason"].map(lambda s: pretty.get(s, s))
    explained = g[g["clinical_reason"] != "allowed"].sort_values("n")
    unexplained = g[g["clinical_reason"] == "allowed"].sort_values("n")
    g = pd.concat([unexplained, explained])
    y = np.arange(len(g))

    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(g) + 1.5))
    ax.barh(y, g["det"], height=0.5, color=BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=3, label="Detected")
    ax.barh(y, g["missed"], left=g["det"], height=0.5, color=GRAY, edgecolor=SURFACE, linewidth=1.2, zorder=3, label="Missed")
    for yi, r in zip(y, g.itertuples()):
        ax.text(r.n + 4, yi, f"{r.det}/{r.n}  ({pct(r.det / r.n, 0)})", va="center", ha="left", fontsize=8.5, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(g["label"])
    ax.set_xlim(0, g["n"].max() * 1.25)
    ax.set_xlabel("Ground-truth infeasible state–action pairs")
    ax.legend(handles=[Patch(color=BLUE, label="Detected"), Patch(color=GRAY, label="Missed")], loc="lower right", ncol=2,
              handlelength=1.0, handleheight=1.0)
    style(ax, grid="x")
    title(ax, "", "Recall by clinical reason")
    ax.set_title("Recall by clinical reason", loc="left", pad=10)
    footnote(fig, "Reason = dominant clinical rule behind the infeasibility mask; 'not explained' = mask marks the pair infeasible but no rule applies")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig5_recall_by_clinical_reason")


def fig_roc(summary, t):
    """Args: summary, t. Saves fig6_roc.pdf/.png."""
    thr = t["sweep_thr"]
    n_r = summary["training_config"]["num_restarts"]
    pts = thr.drop_duplicates(subset=["fpr", "tpr"]).copy()
    pts["k"] = np.where(pts["threshold"] <= 0, 0, np.ceil(pts["threshold"] * n_r - 1e-9)).astype(int)
    pts = pts.sort_values(["fpr", "tpr"])
    curve = pd.concat([pd.DataFrame({"fpr": [0.0], "tpr": [0.0], "k": [n_r + 1]}), pts[["fpr", "tpr", "k"]]]).drop_duplicates(["fpr", "tpr"]).sort_values(["fpr", "tpr"])
    auc = float(np.trapz(curve["tpr"], curve["fpr"]))
    restart = t["restart"]
    operating = float(summary["action_specific_thresholds"]["0"])
    k_op = max(1, int(np.ceil(operating * n_r - 1e-9)))

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 3.6))
    for ax, zoom in ((a, False), (b, True)):
        ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, zorder=1)
        ax.plot(curve["fpr"], curve["tpr"], color=BLUE, linewidth=2, zorder=3, solid_capstyle="round")
        ax.plot(curve["fpr"], curve["tpr"], "o", color=BLUE, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
        ax.plot(restart["fpr"], restart["tpr"], "o", color=ORANGE, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=5)
        style(ax, grid="both")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
    a.set_xlim(-0.02, 1.02); a.set_ylim(-0.02, 1.02)
    title(a, "a", "ROC over vote threshold")
    a.text(0.97, 0.05, f"AUC = {auc:.3f}", ha="right", va="bottom", fontsize=10, color=INK, fontweight="bold")

    inner = curve[curve["k"] > 0]
    xmax = max(0.01, float(max(inner["fpr"].max(), restart["fpr"].max())) * 2.4)
    b.set_xlim(-xmax * 0.06, xmax); b.set_ylim(-0.02, 1.02)
    for r in curve.itertuples():
        if r.k > n_r or r.k == 0 or (r.fpr == 0 and r.tpr == 0):
            continue
        lab = f"≥ {r.k} of {n_r}" if r.k > 0 else "any"
        b.text(r.fpr + xmax * 0.03, r.tpr, lab, va="center", ha="left", fontsize=8.5,
               color=INK if r.k == k_op else INK2, fontweight="bold" if r.k == k_op else "normal")
    title(b, "b", "Zoom on the low-FPR region")
    b.legend(handles=[Line2D([], [], marker="o", color=BLUE, linewidth=2, markersize=6, markeredgecolor=SURFACE, label="Restart vote ensemble"),
                      Line2D([], [], marker="o", color=ORANGE, linewidth=0, markersize=6, markeredgecolor=SURFACE, label="Single restart")],
             loc="center right", handlelength=1.6)
    footnote(fig, f"Vote ensemble: pair flagged if learned in >= k of {n_r} bootstrap restarts  |  bold = operating point (k >= {k_op})")
    fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=2.5)
    save(fig, "fig6_roc")


def fig_lambda(summary, t):
    """Args: summary, t. Saves fig7_lambda_sweep.pdf/.png, or returns None if lambda_sweep.csv is absent."""
    lam = t["lam"]
    if lam is None:
        print("skip fig7 (no lambda_sweep.csv)")
        return
    n_demos = int(summary["n_trajectories"])
    bic = 0.5 * np.log(n_demos)
    floor = 0.03

    def xpos(v):
        """Args: v. Returns: v clamped to the log-scale plotting floor."""
        return np.maximum(np.asarray(v, dtype=float), floor)

    grid = sorted(lam["lambda"].unique())
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 3.4))
    for ax in (a, b):
        ax.set_xscale("log")
        ax.axvline(bic, color=INK2, linewidth=0.9, zorder=2)
        ax.set_xticks([floor, 0.25, 1, 3, 10])
        ax.set_xticklabels(["≈0", "0.25", "1", "3", "10"])
        ax.minorticks_off()
        ax.set_xlim(floor * 0.8, 14)
        ax.set_xlabel(r"Complexity penalty $\lambda$")
        style(ax)

    def lines(ax, col, color, label):
        """Args: ax, col, color, label. Returns: per-lambda mean Series; plots per-restart and mean lines for col on ax."""
        for _, g in lam.groupby("restart"):
            g = g.sort_values("lambda")
            ax.plot(xpos(g["lambda"]), g[col], color=color, alpha=0.22, linewidth=1, zorder=2)
        m = lam.groupby("lambda")[col].mean().reindex(grid)
        ax.plot(xpos(grid), m.values, color=color, linewidth=2, zorder=3, solid_capstyle="round")
        ax.plot(xpos(grid), m.values, "o", color=color, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
        return m

    lines(a, "n_constraints", BLUE, "Constraints")
    a.set_ylabel("Constraints learned")
    a.set_ylim(bottom=0)
    title(a, "a", "Model size")
    lines(b, "tpr", BLUE, "TPR")
    lines(b, "fpr", ORANGE, "FPR")
    b.set_ylim(0, 1.0)
    b.set_ylabel("Rate")
    b.legend(handles=[Line2D([], [], color=BLUE, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, label="True positive rate"),
                      Line2D([], [], color=ORANGE, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, label="False positive rate")],
             loc="upper right", handlelength=1.6)
    title(b, "b", "Detection")
    for ax in (a, b):
        ax.text(bic * 1.06, ax.get_ylim()[1] * 0.02, f"BIC\n{bic:.2f}", ha="left", va="bottom", fontsize=8, color=INK2)
    footnote(fig, "Thick line: mean of restarts; thin lines: individual restarts  |  a constraint is accepted if its log-likelihood gain exceeds lambda")
    fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=2.5)
    save(fig, "fig7_lambda_sweep")


def fig_restarts(summary, t):
    """Args: summary, t. Saves fig8_restart_stability.pdf/.png."""
    r = t["restart"]
    d = t["detect"]
    x = np.arange(len(r))
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 3.3), gridspec_kw={"width_ratios": [1.5, 1]})
    off, bw = 0.2, 0.34
    for j, (col, color, label) in enumerate((("tpr", BLUE, "Recall (TPR)"), ("precision", AQUA, "Precision"))):
        xs = x + (j - 0.5) * 2 * off
        a.bar(xs, r[col], width=bw, color=color, edgecolor=SURFACE, linewidth=0.8, zorder=3)
        for xi, v in zip(xs, r[col]):
            bar_label(a, xi, v, f"{v:.2f}", dy=0.012)
    a.axhline(float(d["tpr"]), color=INK2, linewidth=1, zorder=5)
    a.set_xticks(x); a.set_xticklabels([f"{i + 1}" for i in r["restart"]])
    a.set_xlabel("Bootstrap restart")
    a.set_ylabel("Metric value"); a.set_ylim(0, 1.32)
    a.legend(handles=[Patch(color=BLUE, label="Recall (TPR)"), Patch(color=AQUA, label="Precision"),
                      Line2D([], [], color=INK2, linewidth=1, label=f"Vote-ensemble recall ({float(d['tpr']):.2f})")],
             loc="upper center", ncol=3, handlelength=1.2, handleheight=1.0, fontsize=8, columnspacing=1.0)
    style(a)
    title(a, "a", "Recall and precision per restart")

    b.bar(x, r["n_constraints"], width=0.5, color=GRAY, edgecolor=SURFACE, zorder=3)
    for xi, v in zip(x, r["n_constraints"]):
        bar_label(b, xi, v, f"{int(v)}", dy=0.01 * r["n_constraints"].max())
    b.set_xticks(x); b.set_xticklabels([f"{i + 1}" for i in r["restart"]])
    b.set_xlabel("Bootstrap restart"); b.set_ylabel("Constraints learned")
    b.set_ylim(0, r["n_constraints"].max() * 1.15)
    style(b)
    title(b, "b", "Model size")
    footnote(fig, "Each restart refits on a bootstrap resample of the training patients")
    fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=2.5)
    save(fig, "fig8_restart_stability")


ABLATION_LABELS = {
    "0_joint_path (current)": "Joint EM, path likelihood (baseline)",
    "1_frozen_path": "Frozen reward, path likelihood",
    "2_frozen_local": "Frozen reward, local likelihood",
    "3_frozen_local+rank": "  + discrepancy ranking",
    "4_frozen_local+soft_candidates": "  + soft candidates (ε, η)",
    "5_frozen_local+expand": "  + expand-on-confirm generalization",
    "6_frozen_local+rule": "  + rule-level generalization (adopted)",
    "7_frozen_path+rule": "Frozen, path likelihood + rule",
    "8_frozen_local+rule, no prefilter": "Adopted, safety prefilter off",
    "9_frozen_local, no prefilter": "Frozen local, prefilter off",
    "10_all (local+rule+rank+soft)": "All remedies combined",
}


def fig_ablation(summary, t):
    """Args: summary, t. Saves fig9_ablation.pdf/.png, or returns None if ablation_low_tpr.csv is absent."""
    ab = t["ablation"]
    if ab is None:
        print("skip fig9 (no ablation_low_tpr.csv)")
        return
    ab = ab.copy()
    ab["label"] = ab["config"].map(lambda s: ABLATION_LABELS.get(s, s))
    ab = ab.sort_values("tpr")
    y = np.arange(len(ab))
    colors = [BLUE if c.startswith("6_") else ORANGE if c.startswith("0_") else GRAY for c in ab["config"]]
    fig, ax = plt.subplots(figsize=(7.6, 0.36 * len(ab) + 1.5))
    ax.barh(y, ab["tpr"], height=0.55, color=colors, edgecolor=SURFACE, linewidth=1, zorder=3)
    for yi, r in zip(y, ab.itertuples()):
        ax.text(r.tpr + 0.01, yi, f"TPR {r.tpr:.2f}  ·  FPR {r.fpr:.2f}", va="center", ha="left", fontsize=8.5, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels(ab["label"])
    ax.set_xlim(0, max(ab["tpr"].max() * 1.45, 0.3))
    ax.set_xlabel("True positive rate (mean over restarts, hard vote)")
    ax.legend(handles=[Patch(color=BLUE, label="Adopted default"), Patch(color=ORANGE, label="Baseline"), Patch(color=GRAY, label="Variants")],
              loc="lower right", handlelength=1.0, handleheight=1.0)
    style(ax, grid="x")
    ax.set_title("Ablation of the low-TPR remedies", loc="left", pad=10)
    footnote(fig, "Restart-level metrics (no label-tuned thresholds)  |  safety prefilter on unless stated")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig9_ablation")


def fig_diagnostics(summary, t):
    """Args: summary, t. Saves fig10_low_tpr_diagnostics.pdf/.png, or returns None if diagnostics are absent."""
    dg = t["diag"]
    if dg is None:
        print("skip fig10 (no diagnostics)")
        return
    dp = t["diag_path"]
    n = len(dg)
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(9.6, 3.3), gridspec_kw={"width_ratios": [1.05, 1, 0.85]})

    frac = dg["frac_gt_pairs_in_reachable_states"].to_numpy(float)
    y = np.arange(n)[::-1]
    a.barh(y, frac, height=0.55, color=BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=3)
    a.barh(y, 1 - frac, left=frac, height=0.55, color=GRAY, edgecolor=SURFACE, linewidth=1.2, zorder=3)
    for yi, f in zip(y, frac):
        a.text(f + 0.012, yi, pct(f, 0), va="center", ha="left", fontsize=8, color="white" if False else INK)
    a.set_yticks(y); a.set_yticklabels([f"R{int(r) + 1}" for r in dg["restart"]])
    a.set_xlim(0, 1); a.set_xticks([0, 0.25, 0.5, 0.75, 1]); a.set_xticklabels(["0", "25", "50", "75", "100"])
    a.set_xlabel("Ground-truth infeasible pairs (%)")
    a.legend(handles=[Patch(color=BLUE, label="Reachable from start"), Patch(color=GRAY, label="Unreachable")],
             loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2, handlelength=1.0, handleheight=1.0, columnspacing=1.0, fontsize=8)
    style(a, grid="x")
    title(a, "a", "Where constraints live")

    series = [("Local + rule (adopted)", dg, BLUE)]
    if dp is not None:
        series.insert(0, ("Path, pointwise", dp, ORANGE))
    cats = [("tpr_reachable_states", "Reachable"), ("tpr_unreachable_states", "Unreachable")]
    nser = len(series)
    bw = 0.34 if nser == 2 else 0.5
    rng = np.random.default_rng(0)
    for j, (label, frame, color) in enumerate(series):
        for i, (col, _) in enumerate(cats):
            xi = i + (j - (nser - 1) / 2) * (bw + 0.05)
            vals = frame[col].to_numpy(float)
            b.bar(xi, np.nanmean(vals), width=bw, color=color, edgecolor=SURFACE, linewidth=0.8, zorder=3)
            b.plot(xi + rng.uniform(-0.06, 0.06, len(vals)), vals, "o", color=INK2, markersize=3.5, markeredgecolor=SURFACE, markeredgewidth=0.6, zorder=4)
            bar_label(b, xi, max(np.nanmax(vals), np.nanmean(vals)), f"{np.nanmean(vals):.2f}", dy=0.02)
    b.set_xticks(range(len(cats))); b.set_xticklabels([c[1] for c in cats])
    b.set_xlabel("State reachability under T(s, a)")
    b.set_ylabel("True positive rate"); b.set_ylim(0, 1.15)
    b.legend(handles=[Patch(color=col, label=lab) for lab, _, col in series], loc="upper center", bbox_to_anchor=(0.5, -0.27),
             ncol=1, handlelength=1.0, handleheight=1.0)
    style(b)
    title(b, "b", "Recall by reachability")

    cols = [("mean_pi_unconstrained_detected", "Detected"), ("mean_pi_unconstrained_missed", "Missed")]
    for i, (col, lab) in enumerate(cols):
        vals = dg[col].to_numpy(float)
        c.bar(i, np.nanmean(vals), width=0.5, color=BLUE if i == 0 else GRAY, edgecolor=SURFACE, zorder=3)
        c.plot(i + rng.uniform(-0.07, 0.07, len(vals)), vals, "o", color=INK2, markersize=3.5, markeredgecolor=SURFACE, markeredgewidth=0.6, zorder=4)
        bar_label(c, i, max(np.nanmax(vals), np.nanmean(vals)), f"{np.nanmean(vals):.2f}", dy=0.012)
    c.axhline(0.05, color=INK2, linewidth=0.9, zorder=2)
    c.text(1.98, 0.056, "absorbed", ha="right", va="bottom", fontsize=8, color=INK2)
    c.set_xticks([0, 1]); c.set_xticklabels([l for _, l in cols])
    c.set_xlim(-0.6, 2.0)
    c.set_ylabel(r"Unconstrained $\pi(a\,|\,s)$"); c.set_ylim(0, 0.5)
    style(c)
    title(c, "c", "Reward absorption")
    footnote(fig, "Dots: individual bootstrap restarts  |  (c) mean model probability of the avoided action; absorbed if < 0.05 (<= "
                  f"{100 * float(dg['frac_missed_absorbed_pi_lt_0.05'].max()):.0f}% of missed pairs)")
    fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=2)
    save(fig, "fig10_low_tpr_diagnostics")


def main(results_dir=None) -> list[str]:
    """Args: results_dir. Returns: list of saved figure file paths; runs configure, load, all fig_* functions, and removes stale figure files."""
    configure(results_dir)
    summary, tables = load()
    for fn in (fig_preferences, fig_longitudinal, fig_detection, fig_per_action, fig_reasons, fig_roc, fig_lambda,
               fig_restarts, fig_ablation, fig_diagnostics):
        fn(summary, tables)

    for stale in sorted(OUT.glob("fig*")):
        if stale.suffix in (".png", ".pdf") and str(stale) not in SAVED:
            stale.unlink()
            print("Removed stale figure:", stale.name)
    return list(SAVED)


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else None)
