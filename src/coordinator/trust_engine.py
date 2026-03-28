from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class AgentTrustRecord:
    agent_id: str
    score: float = 1.0           # starts fully trusted
    heartbeat_count: int = 0
    last_seen: float = field(default_factory=time.time)
    history: list = field(default_factory=list)

    def status(self, quarantine_t: float = 0.4, suspicious_t: float = 0.7) -> str:
        if self.score < quarantine_t:
            return "QUARANTINED"
        if self.score < suspicious_t:
            return "SUSPICIOUS"
        return "TRUSTED"


class TrustEngine:
    def __init__(
        self,
        alpha: float = 0.9,
        quarantine_threshold: float = 0.4,
        suspicious_threshold: float = 0.7,
        normalization_ceiling: float = 10.0,
    ):
        self.alpha = alpha
        self.quarantine_threshold = quarantine_threshold
        self.suspicious_threshold = suspicious_threshold
        self.norm_ceiling = normalization_ceiling
        self._records: dict[str, AgentTrustRecord] = {}

    def _get_record(self, agent_id: str) -> AgentTrustRecord:
        if agent_id not in self._records:
            self._records[agent_id] = AgentTrustRecord(agent_id=agent_id)
        return self._records[agent_id]

    def update(self, agent_id: str, divergence: float) -> AgentTrustRecord:
        record = self._get_record(agent_id)
        penalty = min(divergence / self.norm_ceiling, 1.0)
        health = 1.0 - penalty
        old = record.score
        record.score = round(
            max(0.0, min(1.0, self.alpha * old + (1 - self.alpha) * health)), 4
        )
        record.heartbeat_count += 1
        record.last_seen = time.time()
        record.history.append({
            "t": record.last_seen,
            "score": record.score,
            "divergence": round(divergence, 4),
        })
        if len(record.history) > 200:
            record.history = record.history[-200:]
        return record

    def get_all_scores(self) -> dict[str, float]:
        return {aid: r.score for aid, r in self._records.items()}

    def get_record(self, agent_id: str) -> Optional[AgentTrustRecord]:
        return self._records.get(agent_id)

    def is_quarantined(self, agent_id: str) -> bool:
        r = self._records.get(agent_id)
        return r is not None and r.score < self.quarantine_threshold