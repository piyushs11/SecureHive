import json
import os
import sys

import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.detection.semantic_divergence import SemanticDivergenceDetector
from src.privacy.dp_embeddings import gaussian_mechanism, privacy_report


def evaluate(dataset_path: str = "data/raw/attack_dataset.json") -> None:
    with open(dataset_path) as f:
        dataset = json.load(f)

    print(f"\nLoaded {len(dataset)} episodes from {dataset_path}")
    print("=" * 60)

    for epsilon in [0.1, 0.5, 1.0, 5.0, float("inf")]:
        detector = SemanticDivergenceDetector(alert_threshold=3.5)
        y_true, y_pred = [], []

        for episode in dataset:
            raw_emb = np.array(episode["embedding"], dtype=np.float32)

            if epsilon == float("inf"):
                emb = raw_emb
            else:
                emb = gaussian_mechanism(raw_emb, epsilon=epsilon)

            result = detector.score_embedding(episode["agent_id"], emb)
            y_true.append(episode["label"])
            y_pred.append(1 if result["alert"] else 0)

        label = f"ε={epsilon}" if epsilon != float("inf") else "ε=∞ (no DP)"
        auroc = roc_auc_score(y_true, y_pred)
        report = classification_report(
            y_true, y_pred,
            target_names=["benign", "attack"],
            digits=3,
        )
        print(f"\n--- {label} ---")
        print(report)
        print(f"AUROC: {auroc:.4f}")
        if epsilon != float("inf"):
            pr = privacy_report(epsilon)
            print(f"Privacy: σ={pr['sigma']}, level={pr['privacy_level']}")

    print("\n" + "=" * 60)
    print("Evaluation complete. Use these numbers in your report Table 1.")


if __name__ == "__main__":
    evaluate()