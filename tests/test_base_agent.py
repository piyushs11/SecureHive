import sys, os, json, asyncio, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("SHARED_AES_KEY", "a" * 64)

from src.agents.base_agent import BaseAgent


def _make_agent(tmp_path) -> BaseAgent:
    from src.crypto.utils import generate_keypair
    generate_keypair("test_agent", keys_dir=str(tmp_path))
    return BaseAgent(
        agent_id="test_agent",
        role="unit testing",
        coordinator_url="http://localhost:9999",  # nothing running here
        keys_dir=str(tmp_path),
        epsilon=1.0,
        heartbeat_interval=999,
    )


def test_action_log_capped(tmp_path):
    agent = _make_agent(tmp_path)
    for i in range(150):
        agent._log_action(f"action {i}")
    assert len(agent.action_log) == 100


def test_behavioral_summary_no_actions(tmp_path):
    agent = _make_agent(tmp_path)
    summary = agent._build_behavioral_summary()
    assert "no actions" in summary


def test_behavioral_summary_with_actions(tmp_path):
    agent = _make_agent(tmp_path)
    agent._log_action("called database API")
    agent._log_action("returned 5 records")
    summary = agent._build_behavioral_summary()
    assert "called database API" in summary
    assert agent.agent_id in summary


def test_handle_task_logs_action(tmp_path):
    agent = _make_agent(tmp_path)
    asyncio.run(agent._handle_task({"instruction": "fetch records"}))
    assert len(agent.action_log) == 1
    assert "fetch records" in agent.action_log[0]


def test_heartbeat_fails_gracefully_no_coordinator(tmp_path):
    agent = _make_agent(tmp_path)
    agent._log_action("did some work")
    result = asyncio.run(agent.send_heartbeat())
    assert result == {}