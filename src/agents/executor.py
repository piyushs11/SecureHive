import asyncio
from src.agents.base_agent import BaseAgent


class ExecutorAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(
            agent_id="executor",
            role="Executes finalized instructions on external systems",
            **kwargs,
        )

    async def _handle_task(self, task: dict) -> dict:
        instruction = task.get("instruction", "")

        result_msg = (
            f"Executed instruction: {instruction[:80]}. "
            f"Result: SUCCESS."
        )
        self._log_action(result_msg)

        return {
            "agent_id": self.agent_id,
            "executed": instruction[:80],
            "result": "SUCCESS",
            "status": "ok",
        }


if __name__ == "__main__":
    agent = ExecutorAgent(epsilon=5.0, heartbeat_interval=5.0)
    asyncio.run(agent.start(port=8004))