import asyncio
import random
from src.agents.base_agent import BaseAgent


class RetrieverAgent(BaseAgent):

    def __init__(self, **kwargs):
        super().__init__(
            agent_id="retriever",
            role="Fetches relevant knowledge from data sources",
            **kwargs,
        )

    async def _handle_task(self, task: dict) -> dict:
        instruction = task.get("instruction", "")
        n_records = random.randint(3, 15)
        self._log_action(
            f"Fetched {n_records} records for query: {instruction[:60]}"
        )
        return {
            "agent_id": self.agent_id,
            "records_found": n_records,
            "query": instruction,
            "status": "ok",
        }


if __name__ == "__main__":
    agent = RetrieverAgent(epsilon=1.0, heartbeat_interval=10.0)
    asyncio.run(agent.start(port=8002))