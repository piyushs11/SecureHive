import asyncio
from src.agents.base_agent import BaseAgent


class PlannerAgent(BaseAgent):

    def __init__(self, **kwargs):
        super().__init__(
            agent_id="planner",
            role="Decomposes user requests into ordered subtasks",
            **kwargs,
        )

    async def _handle_task(self, task: dict) -> dict:
        instruction = task.get("instruction", "")
        steps = [
            f"1. Retrieve relevant data for: {instruction}",
            f"2. Validate retrieved data against policy",
            f"3. Execute approved action for: {instruction}",
        ]
        plan = " | ".join(steps)
        self._log_action(f"Planned task: {instruction[:60]}")
        return {
            "agent_id": self.agent_id,
            "plan": steps,
            "status": "ok",
        }


if __name__ == "__main__":
    agent = PlannerAgent(epsilon=1.0, heartbeat_interval=10.0)
    asyncio.run(agent.start(port=8001))