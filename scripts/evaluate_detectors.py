import json
import os
import sys
import random
from typing import Any, cast

import numpy as np
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.detection.semantic_divergence import SemanticDivergenceDetector
from src.privacy.dp_embeddings import gaussian_mechanism, privacy_report


def evaluate(
    dataset_path: str = "data/raw/attack_dataset.json",
    results_dir: str = "data/results",
) -> None:
    os.makedirs(results_dir, exist_ok=True)

    with open(dataset_path) as f:
        dataset = json.load(f)

    # Split dataset by label
    benign_episodes  = [e for e in dataset if e["label"] == 0]
    attack_episodes  = [e for e in dataset if e["label"] == 1]

    random.seed(42)
    random.shuffle(benign_episodes)
    n_warmup  = int(len(benign_episodes) * 0.75)
    warmup_ep = benign_episodes[:n_warmup]
    test_benign = benign_episodes[n_warmup:]

    test_set  = test_benign + attack_episodes
    random.shuffle(test_set)

    print(f"\nDataset: {len(dataset)} total episodes")
    print(f"  Warmup (benign only): {len(warmup_ep)}")
    print(f"  Test benign:          {len(test_benign)}")
    print(f"  Test attack:          {len(attack_episodes)}")
    print(f"  Test total:           {len(test_set)}")
    print("=" * 65)

    all_results = {}

    for epsilon in [0.1, 0.5, 1.0, 5.0, float("inf")]:
        label = f"ε={epsilon}" if epsilon != float("inf") else "ε=∞ (no DP, baseline)"

        detector = SemanticDivergenceDetector(
            window_size=50,
            alert_threshold=None,   # adaptive
            warmup_samples=20,
            alert_z_score=3.0,
        )

        for episode in warmup_ep:
            raw_emb = np.array(episode["embedding"], dtype=np.float32)
            emb = raw_emb if epsilon == float("inf") else gaussian_mechanism(raw_emb, epsilon=epsilon)
            detector.score_embedding("executor", emb)

        threshold = detector.get_threshold("executor")
        print(f"\n[{label}] Calibrated threshold: {threshold:.3f}")

        y_true, y_pred, y_score = [], [], []

        for episode in test_set:
            raw_emb = np.array(episode["embedding"], dtype=np.float32)
            emb = raw_emb if epsilon == float("inf") else gaussian_mechanism(raw_emb, epsilon=epsilon)
            result = detector.score_embedding("executor", emb)

            y_true.append(episode["label"])
            y_pred.append(1 if result["alert"] else 0)
            y_score.append(min(result["divergence"] / (threshold * 3), 1.0))

        report = classification_report(
            y_true, y_pred,
            target_names=["benign", "attack"],
            digits=3,
            output_dict=True,
        )
        report_dict = cast(dict[str, dict[str, Any]], report)
        try:
            auroc = roc_auc_score(y_true, y_score)
        except Exception:
            auroc = 0.0

        cm = confusion_matrix(y_true, y_pred)

        print(classification_report(y_true, y_pred, target_names=["benign","attack"], digits=3))
        print(f"AUROC:             {auroc:.4f}")
        print(f"Confusion matrix:  TN={cm[0][0]} FP={cm[0][1]} | FN={cm[1][0]} TP={cm[1][1]}")

        if epsilon != float("inf"):
            pr = privacy_report(epsilon)
            print(f"Privacy:           σ={pr['sigma']}, level={pr['privacy_level']}")

        all_results[str(epsilon)] = {
            "precision_attack":  round(float(report_dict["attack"]["precision"]), 4),
            "recall_attack":     round(float(report_dict["attack"]["recall"]), 4),
            "f1_attack":         round(float(report_dict["attack"]["f1-score"]), 4),
            "precision_benign":  round(float(report_dict["benign"]["precision"]), 4),
            "recall_benign":     round(float(report_dict["benign"]["recall"]), 4),
            "f1_benign":         round(float(report_dict["benign"]["f1-score"]), 4),
            "auroc":             round(auroc, 4),
            "threshold":         round(threshold, 4),
            "confusion_matrix":  cm.tolist(),
        }

    out_path = os.path.join(results_dir, "detection_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*65}")
    print(f"Results saved to {out_path}")
    print("Run scripts/generate_plots.py to generate figures for your report.")


if __name__ == "__main__":
    evaluate(dataset_path="data/raw/attack_dataset_large.json")