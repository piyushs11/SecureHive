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
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.detection.semantic_divergence import SemanticDivergenceDetector
from src.privacy.dp_embeddings import gaussian_mechanism, privacy_report


AGENT_ROLES = ["planner", "retriever", "policy_checker", "executor"]


def evaluate(
    dataset_path: str = "data/raw/attack_dataset_large.json",
    results_dir: str = "data/results",
) -> None:
    os.makedirs(results_dir, exist_ok=True)

    with open(dataset_path) as f:
        dataset = json.load(f)

    print(f"\nDataset: {len(dataset)} total episodes")
    print(f"Agents: {AGENT_ROLES}")
    print("=" * 65)

    all_results: dict = {"dataset_size": len(dataset)}

    for epsilon in [0.1, 0.5, 1.0, 5.0, float("inf")]:
        label = f"ε={epsilon}" if epsilon != float("inf") else "ε=∞ (no DP, baseline)"
        print(f"\n{'='*65}")
        print(f"[{label}]")

        agg_y_true: list[int] = []
        agg_y_pred: list[int] = []
        agg_y_score: list[float] = []

        for agent_id in AGENT_ROLES:
            agent_eps = [e for e in dataset if e["agent_id"] == agent_id]
            benign_ep = [e for e in agent_eps if e["label"] == 0]
            attack_ep = [e for e in agent_eps if e["label"] == 1]

            if len(benign_ep) < 50:
                print(f"  {agent_id}: insufficient data, skipping")
                continue

            random.seed(42)
            random.shuffle(benign_ep)
            n_warmup    = int(len(benign_ep) * 0.75)
            warmup_ep   = benign_ep[:n_warmup]
            test_benign = benign_ep[n_warmup:]

            detector = SemanticDivergenceDetector(
                window_size=50,
                alert_threshold=None,   
                warmup_samples=20,
                alert_z_score=3.0,
            )

            for ep in warmup_ep:
                raw = np.array(ep["embedding"], dtype=np.float32)
                emb = (raw if epsilon == float("inf")
                       else gaussian_mechanism(raw, epsilon=epsilon))
                detector.score_embedding(agent_id, emb)

            threshold = detector.get_threshold(agent_id)
            print(f"  [{agent_id}] threshold={threshold:.3f}  "
                  f"warmup={len(warmup_ep)}  "
                  f"test_benign={len(test_benign)}  "
                  f"attacks={len(attack_ep)}")

            # Evaluate on held-out benign 
            for ep in test_benign:
                raw = np.array(ep["embedding"], dtype=np.float32)
                emb = (raw if epsilon == float("inf")
                       else gaussian_mechanism(raw, epsilon=epsilon))
                result = detector.score_embedding(agent_id, emb)
                agg_y_true.append(0)
                agg_y_pred.append(1 if result["alert"] else 0)
                agg_y_score.append(
                    min(result["divergence"] / (threshold * 3 + 1e-6), 1.0)
                )

            # Evaluate on ALL attacks for this agent 
            for ep in attack_ep:
                raw = np.array(ep["embedding"], dtype=np.float32)
                emb = (raw if epsilon == float("inf")
                       else gaussian_mechanism(raw, epsilon=epsilon))
                result = detector.score_embedding(agent_id, emb)
                agg_y_true.append(1)
                agg_y_pred.append(1 if result["alert"] else 0)
                agg_y_score.append(
                    min(result["divergence"] / (threshold * 3 + 1e-6), 1.0)
                )

        # Aggregate metrics across all agents
        report = classification_report(
            agg_y_true,
            agg_y_pred,
            target_names=["benign", "attack"],
            digits=3,
            output_dict=True,
        )
        report_dict = cast(dict[str, dict[str, Any]], report)

        try:
            auroc = roc_auc_score(agg_y_true, agg_y_score)
        except Exception:
            auroc = 0.0

        cm = confusion_matrix(agg_y_true, agg_y_pred)

        print()
        print(classification_report(
            agg_y_true, agg_y_pred,
            target_names=["benign", "attack"], digits=3,
        ))
        print(f"AUROC:             {auroc:.4f}")
        print(
            f"Confusion matrix:  "
            f"TN={cm[0][0]} FP={cm[0][1]} | FN={cm[1][0]} TP={cm[1][1]}"
        )
        if epsilon != float("inf"):
            pr = privacy_report(epsilon)
            print(f"Privacy:           σ={pr['sigma']}, level={pr['privacy_level']}")

        eps_key = str(epsilon) if epsilon != float("inf") else "inf"
        all_results[eps_key] = {
            "precision_attack": round(float(report_dict["attack"]["precision"]), 4),
            "recall_attack":    round(float(report_dict["attack"]["recall"]), 4),
            "f1_attack":        round(float(report_dict["attack"]["f1-score"]), 4),
            "precision_benign": round(float(report_dict["benign"]["precision"]), 4),
            "recall_benign":    round(float(report_dict["benign"]["recall"]), 4),
            "f1_benign":        round(float(report_dict["benign"]["f1-score"]), 4),
            "auroc":            round(auroc, 4),
            "confusion_matrix": cm.tolist(),
        }

    out_path = os.path.join(results_dir, "detection_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*65}")
    print(f"Results saved to {out_path}")
    print("Run scripts/generate_plots.py to generate figures for your report.")


if __name__ == "__main__":
    evaluate(dataset_path="data/raw/attack_dataset_large.json")