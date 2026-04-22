import json
import os
import sys
import random

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.detection.semantic_divergence import SemanticDivergenceDetector
from src.privacy.dp_embeddings import gaussian_mechanism


def analyze_fairness(
    dataset_path: str = "data/raw/attack_dataset_large.json",
    epsilon: float = 1.0,
    results_dir: str = "data/results",
) -> None:
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(dataset_path):
        dataset_path = "data/raw/attack_dataset.json"
        print(f"[fairness] Large dataset not found, using {dataset_path}")

    with open(dataset_path) as f:
        dataset = json.load(f)

    agents = list(set(e["agent_id"] for e in dataset))
    print(f"\nFairness Analysis  (ε={epsilon})")
    print(f"Agents analyzed: {agents}")
    print("=" * 60)

    results = {}

    for agent_id in sorted(agents):
        agent_eps = [e for e in dataset if e["agent_id"] == agent_id]
        benign_ep = [e for e in agent_eps if e["label"] == 0]
        attack_ep = [e for e in agent_eps if e["label"] == 1]

        if len(benign_ep) < 50:
            print(f"  {agent_id}: insufficient data (skipping)")
            continue

        random.seed(42)
        random.shuffle(benign_ep)
        n_warmup  = int(len(benign_ep) * 0.75)
        warmup_ep = benign_ep[:n_warmup]
        test_benign = benign_ep[n_warmup:]

        detector = SemanticDivergenceDetector(
            window_size=50, alert_threshold=None,
            warmup_samples=20, alert_z_score=3.0,
        )

        for ep in warmup_ep:
            emb = np.array(ep["embedding"], dtype=np.float32)
            if epsilon != float("inf"):
                emb = gaussian_mechanism(emb, epsilon=epsilon)
            detector.score_embedding(agent_id, emb)

        threshold = detector.get_threshold(agent_id)

        fp_count = 0
        for ep in test_benign:
            emb = np.array(ep["embedding"], dtype=np.float32)
            if epsilon != float("inf"):
                emb = gaussian_mechanism(emb, epsilon=epsilon)
            result = detector.score_embedding(agent_id, emb)
            if result["alert"]:
                fp_count += 1

        fpr = fp_count / len(test_benign) if test_benign else 0.0

        tp_count = 0
        for ep in attack_ep[:len(test_benign)]:  # match sample size
            emb = np.array(ep["embedding"], dtype=np.float32)
            if epsilon != float("inf"):
                emb = gaussian_mechanism(emb, epsilon=epsilon)
            result = detector.score_embedding(agent_id, emb)
            if result["alert"]:
                tp_count += 1

        tpr = tp_count / min(len(attack_ep), len(test_benign)) if attack_ep else 0.0

        results[agent_id] = {
            "n_warmup":         len(warmup_ep),
            "n_test_benign":    len(test_benign),
            "n_attacks_tested": min(len(attack_ep), len(test_benign)),
            "threshold":        round(threshold, 4),
            "false_positive_rate": round(fpr, 4),
            "true_positive_rate":  round(tpr, 4),
        }

        print(f"\n  {agent_id.upper():20s}")
        print(f"    Threshold:           {threshold:.3f}")
        print(f"    False Positive Rate: {fpr:.3f}  ({fp_count}/{len(test_benign)} benign flagged)")
        print(f"    True Positive Rate:  {tpr:.3f}  ({tp_count}/{min(len(attack_ep),len(test_benign))} attacks caught)")

    fprs = [v["false_positive_rate"] for v in results.values()]
    max_disparity = max(fprs) - min(fprs) if len(fprs) > 1 else 0.0

    print(f"\n{'='*60}")
    print(f"Fairness summary (ε={epsilon}):")
    print(f"  FPR range: {min(fprs):.3f} – {max(fprs):.3f}")
    print(f"  Max disparity: {max_disparity:.3f}")
    if max_disparity < 0.05:
        print("  Assessment: FAIR — FPR is consistent across agent roles")
    elif max_disparity < 0.15:
        print("  Assessment: MODERATE — minor FPR variation across agents")
    else:
        print("  Assessment: UNFAIR — significant FPR disparity; per-agent baseline partially compensates")

    results["_meta"] = {
        "epsilon": epsilon,
        "max_fpr_disparity": round(max_disparity, 4),
        "assessment": "fair" if max_disparity < 0.05 else "moderate" if max_disparity < 0.15 else "unfair",
    }

    out_path = os.path.join(results_dir, "fairness_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    for eps in [1.0, 5.0]:
        analyze_fairness(epsilon=eps)