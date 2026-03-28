import sys, os, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.coordinator.trust_engine import TrustEngine

def test_trust_starts_at_one():
    engine = TrustEngine()
    r = engine.update("planner", divergence=0.0)
    assert r.score == pytest.approx(1.0, abs=0.01)

def test_sustained_attack_triggers_quarantine():
    engine = TrustEngine(alpha=0.9, quarantine_threshold=0.4)
    for _ in range(15):
        engine.update("executor", divergence=100.0)

    r = engine.get_record("executor")
    assert r is not None, "Record should exist after sending heartbeats"
    assert r.score < 0.4
    assert r.status() == "QUARANTINED"

def test_recovery_is_slow():
    engine = TrustEngine(alpha=0.9)
    for _ in range(12):
        engine.update("retriever", divergence=100.0)

    r = engine.get_record("retriever")
    assert r is not None
    assert r.score < 0.4

    for _ in range(5):
        engine.update("retriever", divergence=0.0)

    r = engine.get_record("retriever")
    assert r is not None
    assert r.score < 0.7

def test_all_scores_returned():
    engine = TrustEngine()
    engine.update("planner", divergence=0.5)
    engine.update("executor", divergence=8.0)
    scores = engine.get_all_scores()
    assert "planner" in scores and "executor" in scores