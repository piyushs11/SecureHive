import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')   
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_DIR = "data/results"
PLOTS_DIR   = "data/results/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

COLORS = {
    "trusted":     "#1D9E75",
    "suspicious":  "#EF9F27",
    "quarantined": "#E24B4A",
    "purple":      "#7F77DD",
    "blue":        "#378ADD",
    "gray":        "#888780",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size":   11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def load_detection_results():
    path = os.path.join(RESULTS_DIR, "detection_results.json")
    if not os.path.exists(path):
        print(f"[plots] {path} not found. Run evaluate_detectors.py first.")
        return None
    with open(path) as f:
        return json.load(f)


def load_fairness_results():
    path = os.path.join(RESULTS_DIR, "fairness_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def plot_privacy_utility(results):
    epsilons  = [0.1, 0.5, 1.0, 5.0]
    eps_labels = ["0.1", "0.5", "1.0", "5.0"]

    f1_attack   = [results[str(e)]["f1_attack"]   for e in epsilons]
    f1_benign   = [results[str(e)]["f1_benign"]   for e in epsilons]
    auroc       = [results[str(e)]["auroc"]        for e in epsilons]

    no_dp = results.get("inf", {})
    if no_dp:
        f1_attack.append(no_dp["f1_attack"])
        f1_benign.append(no_dp["f1_benign"])
        auroc.append(no_dp["auroc"])
        eps_labels.append("∞\n(no DP)")
        x = list(range(len(eps_labels)))
    else:
        x = list(range(len(eps_labels)))

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(x[:len(f1_attack)], f1_attack, 'o-', color=COLORS["quarantined"],
            linewidth=2, markersize=7, label="Attack F1")
    ax.plot(x[:len(f1_benign)], f1_benign, 's-', color=COLORS["trusted"],
            linewidth=2, markersize=7, label="Benign F1")
    ax.plot(x[:len(auroc)],     auroc,      '^--', color=COLORS["purple"],
            linewidth=1.5, markersize=6, label="AUROC")

    ax.axhline(0.5, color=COLORS["gray"], linewidth=0.8, linestyle=":", alpha=0.7,
               label="Random baseline (0.5)")

    ax.set_xticks(x[:len(eps_labels)])
    ax.set_xticklabels(eps_labels)
    ax.set_xlabel("Differential Privacy Budget  ε  (higher = less private)", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Figure 1: Privacy-Utility Trade-off Curve\n"
                 "PrivAgent-TrustShield · Semantic Divergence Detector", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    ax.axvspan(-0.5, 1.5, alpha=0.04, color=COLORS["trusted"], label="_")
    ax.axvspan(1.5, len(x)-0.5, alpha=0.04, color=COLORS["suspicious"], label="_")
    ax.text(0.5, 0.02, "Strong privacy", ha="center", fontsize=9, color=COLORS["trusted"])
    ax.text(2.5, 0.02, "Moderate/Weak privacy", ha="center", fontsize=9, color=COLORS["suspicious"])

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "privacy_utility_curve.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[plots] Saved: {out}")


def plot_precision_recall(results):
    epsilons   = [0.1, 0.5, 1.0, 5.0]
    eps_labels = ["ε=0.1", "ε=0.5", "ε=1.0", "ε=5.0"]

    prec_attack = [results[str(e)]["precision_attack"] for e in epsilons]
    rec_attack  = [results[str(e)]["recall_attack"]    for e in epsilons]

    x   = np.arange(len(epsilons))
    w   = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, prec_attack, w, label="Precision (attack)", color=COLORS["purple"])
    ax.bar(x + w/2, rec_attack,  w, label="Recall (attack)",    color=COLORS["quarantined"])

    ax.set_xticks(x)
    ax.set_xticklabels(eps_labels)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_title("Figure 2: Attack Detection Precision and Recall\nacross Privacy Budgets", fontsize=12)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for i, (p, r) in enumerate(zip(prec_attack, rec_attack)):
        ax.text(i - w/2, p + 0.02, f"{p:.2f}", ha="center", fontsize=9)
        ax.text(i + w/2, r + 0.02, f"{r:.2f}", ha="center", fontsize=9)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "precision_recall_bars.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[plots] Saved: {out}")


def plot_fairness(fairness):
    if not fairness:
        print("[plots] Fairness results not found, skipping Figure 3.")
        return

    agents = [k for k in fairness if not k.startswith("_")]
    fprs   = [fairness[a]["false_positive_rate"] for a in agents]
    tprs   = [fairness[a]["true_positive_rate"]  for a in agents]

    x  = np.arange(len(agents))
    w  = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - w/2, fprs, w, label="False Positive Rate (benign flagged)",
                   color=COLORS["suspicious"])
    bars2 = ax.bar(x + w/2, tprs, w, label="True Positive Rate (attacks caught)",
                   color=COLORS["trusted"])

    ax.axhline(0.05, color=COLORS["gray"], linewidth=1, linestyle="--", alpha=0.7,
               label="Fairness threshold (FPR ≤ 0.05)")

    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("_", "\n") for a in agents])
    ax.set_ylabel("Rate", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_title("Figure 3: Fairness Analysis — Per-Agent False Positive and True Positive Rates\n"
                 "(ε=1.0,  lower FPR = more fair)", fontsize=12)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "fairness_per_agent.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[plots] Saved: {out}")


def plot_trust_timeline():
    alpha = 0.9
    norm_ceiling = 100.0

    def update(score, divergence):
        penalty = min(divergence / norm_ceiling, 1.0)
        return round(max(0.0, min(1.0, alpha * score + (1 - alpha) * (1 - penalty))), 4)

    beats = 60
    scores_executor = []
    scores_planner  = []
    s_exec = 1.0
    s_plan = 1.0

    for i in range(beats):
        if i < 25:
            div_exec = np.random.uniform(18, 22)   # normal DP noise range
        elif i < 40:
            div_exec = np.random.uniform(150, 350) # attack divergence
        else:
            div_exec = np.random.uniform(18, 22)   # recovery

        # Planner: always clean — never attacked
        div_plan = np.random.uniform(17, 23)

        s_exec = update(s_exec, div_exec)
        s_plan = update(s_plan, div_plan)
        scores_executor.append(s_exec)
        scores_planner.append(s_plan)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = list(range(1, beats + 1))

    ax.plot(x, scores_executor, linewidth=2.5, color=COLORS["quarantined"],
            label="Executor (attack target)", zorder=3)
    ax.plot(x, scores_planner,  linewidth=2, color=COLORS["trusted"],
            label="Planner (not attacked)", linestyle="--", zorder=3)

    ax.axhline(0.7, color=COLORS["trusted"],     linewidth=1,   linestyle=":",  alpha=0.8)
    ax.axhline(0.4, color=COLORS["quarantined"],  linewidth=1,   linestyle=":",  alpha=0.8)
    ax.text(61, 0.72, "trusted (0.7)",     fontsize=9, color=COLORS["trusted"])
    ax.text(61, 0.42, "quarantine (0.4)",  fontsize=9, color=COLORS["quarantined"])

    ax.axvspan(25, 40, alpha=0.08, color=COLORS["quarantined"])
    ax.text(32.5, 0.02, "Attack window\n(15 injected heartbeats)",
            ha="center", fontsize=9, color=COLORS["quarantined"])

    ax.set_xlabel("Heartbeat number  (10 s intervals)", fontsize=12)
    ax.set_ylabel("Trust score", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, 68)
    ax.set_title("Figure 4: Trust Score Timeline — Prompt Injection Attack on Executor\n"
                 "Only the attacked agent degrades; Planner remains trusted throughout", fontsize=12)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "trust_score_timeline.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[plots] Saved: {out}")


if __name__ == "__main__":
    print("Generating all report figures...\n")

    det = load_detection_results()
    if det:
        plot_privacy_utility(det)
        plot_precision_recall(det)
    else:
        print("[plots] Run evaluate_detectors.py first to generate detection_results.json")

    fair = load_fairness_results()
    plot_fairness(fair)
    plot_trust_timeline()   # always works, uses simulation

    print(f"\nAll figures saved to {PLOTS_DIR}/")
    print("Add these to your final report.")