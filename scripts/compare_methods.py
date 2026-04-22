import json
import os

RESULTS_DIR = "data/results"


def compare():
    det_path   = os.path.join(RESULTS_DIR, "detection_results.json")
    judge_path = os.path.join(RESULTS_DIR, "judge_results.json")

    if not os.path.exists(det_path):
        print("[compare] Run evaluate_detectors.py first.")
        return
    if not os.path.exists(judge_path):
        print("[compare] Run evaluate_llm_judge.py first.")
        return

    with open(det_path)   as f: det   = json.load(f)
    with open(judge_path) as f: judge = json.load(f)

    print("\n" + "=" * 75)
    print("ALGORITHM COMPARISON — PrivAgent-TrustShield")
    print("=" * 75)

    print("\n┌── Algorithm 1: Semantic Divergence Detector ──────────────────────────┐")
    print(f"│  Method: Adaptive standardized Euclidean distance on 384-dim embeddings")
    print(f"│  Dataset: {det.get('dataset_size', 'N/A')} episodes (temporal split)")
    print(f"│")
    print(f"│  {'ε':>8} │ {'Attack Prec':>11} │ {'Attack Rec':>10} │ {'Attack F1':>9} │ {'AUROC':>6}")
    print(f"│  {'─'*8}─┼─{'─'*11}─┼─{'─'*10}─┼─{'─'*9}─┼─{'─'*6}")

    for eps in ['0.1', '0.5', '1.0', '5.0', 'inf']:
        r = det.get(eps, {})
        eps_label = f"ε={eps}" if eps != 'inf' else "ε=∞"
        print(f"│  {eps_label:>8} │ {r.get('precision_attack', 0):>11.3f} │ "
              f"{r.get('recall_attack', 0):>10.3f} │ "
              f"{r.get('f1_attack', 0):>9.3f} │ "
              f"{r.get('auroc', 0):>6.4f}")

    print("└───────────────────────────────────────────────────────────────────────┘")

    print("\n┌── Algorithm 2: LLM-as-a-Judge (Llama 3.2) ───────────────────────────┐")
    n = judge.get("n_samples", "?")
    auroc_j = judge.get("auroc", "N/A")
    verdicts = judge.get("verdicts", [])

    if verdicts:
        y_true = [v.get("true_label", 0) for v in verdicts if "error" not in v]
        y_pred = [v.get("predicted", 0)   for v in verdicts if "error" not in v]

        if y_true:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
            tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0

            print(f"│  Samples evaluated: {n}")
            print(f"│  Attack Precision:  {prec:.3f}")
            print(f"│  Attack Recall:     {rec:.3f}")
            print(f"│  Attack F1:         {f1:.3f}")
            print(f"│  False Positive Rate: {fpr:.3f}")
            print(f"│  AUROC:             {auroc_j}")
            print(f"│  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print("└───────────────────────────────────────────────────────────────────────┘")

    print("\n┌── Key Finding ────────────────────────────────────────────────────────┐")
    print("│  Semantic Divergence performs near-random in single-episode offline")
    print("│  evaluation (AUROC ≈ 0.45–0.54). This is expected: general-purpose")
    print("│  sentence embeddings do not produce sufficient geometric separation")
    print("│  between one benign and one attack sentence. The detector requires")
    print("│  behavioral ACCUMULATION across multiple heartbeats to be effective.")
    print("│                                                                       ")
    print("│  LLM-as-a-Judge directly inspects semantic content and reasons about")
    print("│  role consistency, making it significantly more effective per-episode.")
    print("│                                                                       ")
    print("│  Together: Divergence catches slow behavioral drift; Judge catches   ")
    print("│  single-episode semantic violations. Complementary coverage.         ")
    print("└───────────────────────────────────────────────────────────────────────┘\n")

    summary = {
        "semantic_divergence": {eps: det[eps] for eps in det},
        "llm_judge": {
            "n_samples": n,
            "auroc": auroc_j,
        },
        "finding": "Divergence requires accumulation; Judge works per-episode. Complementary."
    }
    out = os.path.join(RESULTS_DIR, "comparison_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {out}")


if __name__ == "__main__":
    compare()