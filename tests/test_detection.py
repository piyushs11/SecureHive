import sys, os
import numpy as np
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.detection.semantic_divergence import SemanticDivergenceDetector


def _fill_baseline(detector: SemanticDivergenceDetector, agent_id: str, n: int = 30):
    base = np.ones(384) * 0.5
    for _ in range(n):
        detector.score_embedding(agent_id, base + np.random.randn(384) * 0.01)


def test_warmup_returns_zero_divergence():
    det = SemanticDivergenceDetector()
    result = det.score_embedding("agent_a", np.ones(384))
    assert result["divergence"] == 0.0
    assert result["alert"] is False
    assert "warming_up" in result["reason"]


def test_normal_behavior_not_flagged():
    np.random.seed(42)
    det = SemanticDivergenceDetector(alert_threshold=100.0)
    _fill_baseline(det, "agent_b", n=30)

    normal_emb = np.ones(384) * 0.5 + np.random.randn(384) * 0.01
    result = det.score_embedding("agent_b", normal_emb)
    assert result["alert"] is False


def test_attack_embedding_flagged():
    np.random.seed(0)
    det = SemanticDivergenceDetector(alert_threshold=3.5)
    _fill_baseline(det, "agent_c", n=30)

    attack_emb = np.ones(384) * -50.0
    result = det.score_embedding("agent_c", attack_emb)
    assert result["alert"] is True
    assert result["divergence"] > 3.5


def test_separate_baselines_per_agent():
    np.random.seed(1)
    det = SemanticDivergenceDetector()

    _fill_baseline(det, "agent_d", n=30)

    base_e = np.ones(384) * -0.5
    for _ in range(30):
        det.score_embedding("agent_e", base_e + np.random.randn(384) * 0.01)

    result_d = det.score_embedding("agent_d", np.ones(384) * 0.5)
    result_e = det.score_embedding("agent_e", np.ones(384) * 0.5)
    assert result_d["divergence"] < result_e["divergence"]


def test_reset_clears_baseline():
    det = SemanticDivergenceDetector()
    _fill_baseline(det, "agent_f", n=20)
    assert det.baseline_size("agent_f") > 0
    det.reset_agent("agent_f")
    assert det.baseline_size("agent_f") == 0