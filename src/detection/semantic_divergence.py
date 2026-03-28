from collections import deque

import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticDivergenceDetector:

    def __init__(self, window_size: int = 50, alert_threshold: float = 3.5):
        self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self._history: dict[str, deque] = {}

    def _get_history(self, agent_id: str) -> deque:
        if agent_id not in self._history:
            self._history[agent_id] = deque(maxlen=self.window_size)
        return self._history[agent_id]

    def score_text(self, agent_id: str, text: str) -> dict:

        embedding: np.ndarray = np.array(
            self._encoder.encode(text), dtype=np.float32
        )
        return self.score_embedding(agent_id, embedding)

    def score_embedding(self, agent_id: str, embedding: np.ndarray) -> dict:

        history = self._get_history(agent_id)

        if len(history) < 10:
            history.append(embedding)
            return {
                "divergence": 0.0,
                "alert": False,
                "reason": f"warming_up ({len(history)}/10 samples)",
            }

        matrix = np.array(history)            # shape: (N, 384)
        mean = matrix.mean(axis=0)            # per-dimension baseline mean

        variances = matrix.var(axis=0) + 1e-6

        delta = embedding - mean

        dist = float(np.sqrt(np.sum((delta ** 2) / variances)))

        history.append(embedding)

        alerted = dist > self.alert_threshold
        return {
            "divergence": round(dist, 4),
            "alert": alerted,
            "reason": (
                f"divergence={dist:.2f} > threshold={self.alert_threshold}"
                if alerted
                else "within normal range"
            ),
        }

    def reset_agent(self, agent_id: str) -> None:
        if agent_id in self._history:
            self._history[agent_id].clear()

    def baseline_size(self, agent_id: str) -> int:
        return len(self._history.get(agent_id, []))