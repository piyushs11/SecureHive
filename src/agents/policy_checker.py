import asyncio
from src.agents.base_agent import BaseAgent

BLOCKED_KEYWORDS = [
    "exfiltrate", "override", "admin_override", "disable_security",
    "bypass", "ignore all", "maintenance mode", "reveal api key",
]


class PolicyCheckerAgent(BaseAgent):

    def __init__(self, **kwargs):
        super().__init__(
            agent_id="policy_checker",
            role="Validates agent actions against security policy",
            **kwargs,
        )

    async def _handle_task(self, task: dict) -> dict:
        instruction = task.get("instruction", "").lower()
        violations = [kw for kw in BLOCKED_KEYWORDS if kw in instruction]

        if violations:
            self._log_action(
                f"BLOCKED action — policy violation: {violations}"
            )
            return {
                "agent_id": self.agent_id,
                "approved": False,
                "violations": violations,
                "status": "blocked",
            }

        self._log_action(f"Approved action: {task.get('instruction', '')[:60]}")
        return {
            "agent_id": self.agent_id,
            "approved": True,
            "violations": [],
            "status": "ok",
        }


if __name__ == "__main__":
    agent = PolicyCheckerAgent(epsilon=1.0, heartbeat_interval=10.0)
    asyncio.run(agent.start(port=8003))