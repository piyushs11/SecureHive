from collections import deque

import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticDivergenceDetector:

    def __init__(
        self,
        window_size: int = 50,
        alert_threshold: float | None = None,   
        warmup_samples: int = 20,
        alert_z_score: float = 3.0,
    ):
        self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.window_size = window_size
        self.fixed_threshold = alert_threshold   
        self.warmup_samples = warmup_samples
        self.alert_z_score = alert_z_score

        self._history: dict[str, deque] = {}
        self._warmup_distances: dict[str, list[float]] = {}
        self._adaptive_threshold: dict[str, float] = {}

    def _get_history(self, agent_id: str) -> deque:
        if agent_id not in self._history:
            self._history[agent_id] = deque(maxlen=self.window_size)
        return self._history[agent_id]

    def _get_threshold(self, agent_id: str) -> float:
        if self.fixed_threshold is not None:
            return self.fixed_threshold
        return self._adaptive_threshold.get(agent_id, float("inf"))

    def score_text(self, agent_id: str, text: str) -> dict:
        embedding: np.ndarray = np.array(
            self._encoder.encode(text), dtype=np.float32
        )
        return self.score_embedding(agent_id, embedding)

    def score_embedding(self, agent_id: str, embedding: np.ndarray) -> dict:

        history = self._get_history(agent_id)
        warmup = self._warmup_distances.setdefault(agent_id, [])

        if len(history) < self.warmup_samples:
            history.append(embedding)
            return {
                "divergence": 0.0,
                "alert": False,
                "reason": f"warming_up ({len(history)}/{self.warmup_samples} samples)",
            }

        if agent_id not in self._adaptive_threshold:
            matrix = np.array(history)              # shape: (warmup_samples, 384)
            mean = matrix.mean(axis=0)
            variances = matrix.var(axis=0) + 1e-6

            warmup_dists = []
            for sample in matrix:
                delta = sample - mean
                d = float(np.sqrt(np.sum((delta ** 2) / variances)))
                warmup_dists.append(d)

            dist_mean = float(np.mean(warmup_dists))
            dist_std  = float(np.std(warmup_dists)) + 1e-6

            adaptive = dist_mean + self.alert_z_score * dist_std
            self._adaptive_threshold[agent_id] = round(adaptive, 4)
            print(
                f"[detector] {agent_id}: threshold calibrated to "
                f"{adaptive:.2f} (mean={dist_mean:.2f}, std={dist_std:.2f})"
            )

        matrix = np.array(history)
        mean = matrix.mean(axis=0)
        variances = matrix.var(axis=0) + 1e-6

        delta = embedding - mean
        dist = float(np.sqrt(np.sum((delta ** 2) / variances)))

        history.append(embedding)

        threshold = self._get_threshold(agent_id)
        alerted = dist > threshold

        return {
            "divergence": round(dist, 4),
            "alert": alerted,
            "threshold": round(threshold, 4),
            "reason": (
                f"divergence={dist:.2f} > threshold={threshold:.2f} — ANOMALY"
                if alerted
                else f"divergence={dist:.2f} within threshold={threshold:.2f}"
            ),
        }

    def reset_agent(self, agent_id: str) -> None:
        self._history.pop(agent_id, None)
        self._warmup_distances.pop(agent_id, None)
        self._adaptive_threshold.pop(agent_id, None)

    def baseline_size(self, agent_id: str) -> int:
        return len(self._history.get(agent_id, []))

    def get_threshold(self, agent_id: str) -> float:
        return self._get_threshold(agent_id)