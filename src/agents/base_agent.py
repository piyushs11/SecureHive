import asyncio
import json
import os
import time
import numpy as np
import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from src.crypto.utils import aes_encrypt, sign_payload, load_private_key
from src.privacy.dp_embeddings import gaussian_mechanism

_key_hex = os.environ.get("SHARED_AES_KEY", "")
if not _key_hex:
    import warnings
    warnings.warn("SHARED_AES_KEY not set — using test default. Never do this in production.")
    _key_hex = "a" * 64
SHARED_AES_KEY = bytes.fromhex(_key_hex)

class TaskRequest(BaseModel):
    instruction: str
    task_id: str = ""


class BaseAgent:

    def __init__(
        self,
        agent_id: str,
        role: str,
        coordinator_url: str | None = None,
        keys_dir: str = "./keys",
        epsilon: float = 1.0,
        heartbeat_interval: float = 10.0,
    ):
        self.agent_id = agent_id
        self.role = role
        self.coordinator_url = coordinator_url if coordinator_url is not None else os.getenv(
            "COORDINATOR_URL", "http://localhost:8000"
    )
        self.epsilon = epsilon
        self.heartbeat_interval = heartbeat_interval

        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

        self.private_key = load_private_key(f"{keys_dir}/{agent_id}_private.pem")

        self.action_log: list[str] = []

        self.app = FastAPI(title=f"Agent: {agent_id}")
        self._register_routes()


    def _register_routes(self):
        @self.app.get("/health")
        def health():
            return {
                "agent_id": self.agent_id,
                "role": self.role,
                "status": "alive",
                "action_count": len(self.action_log),
            }

        @self.app.post("/task")
        async def receive_task(task: TaskRequest):
            result = await self._handle_task(task.dict())
            return result


    async def _handle_task(self, task: dict) -> dict:
        output = (
            f"Agent {self.agent_id} ({self.role}): "
            f"processed [{task.get('instruction', '')}]"
        )
        self._log_action(output)
        return {"agent_id": self.agent_id, "output": output, "status": "ok"}

    def _log_action(self, text: str):
        self.action_log.append(text)
        if len(self.action_log) > 100:
            self.action_log = self.action_log[-100:]


    def _build_behavioral_summary(self) -> str:
        recent = self.action_log[-10:] if self.action_log else []
        if not recent:
            return f"Agent {self.agent_id} ({self.role}): no actions recorded yet."
        actions = "; ".join(recent)
        return f"Agent {self.agent_id} ({self.role}). Recent actions: {actions}"


    async def send_heartbeat(self) -> dict:
        summary = self._build_behavioral_summary()

        raw_embedding: np.ndarray = np.array(self.encoder.encode(summary), dtype=np.float32)


        protected_emb: np.ndarray = gaussian_mechanism(
            raw_embedding, epsilon=self.epsilon
        )

        payload = {
            "agent_id": self.agent_id,
            "timestamp": time.time(),
            "epsilon_used": self.epsilon,
            "dp_embedding": protected_emb.tolist(),
            "action_count": len(self.action_log),
        }

        payload_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        signature = sign_payload(payload_bytes, self.private_key)

        inner = json.dumps(
            {"payload": payload, "signature": signature}
        ).encode("utf-8")
        envelope = aes_encrypt(inner, SHARED_AES_KEY)

        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                resp = await client.post(
                    f"{self.coordinator_url}/telemetry",
                    json={"agent_id": self.agent_id, "envelope": envelope},
                )
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as e:
                print(f"[{self.agent_id}] Heartbeat rejected: {e.response.status_code}")
                return {}
            except Exception as e:
                print(f"[{self.agent_id}] Heartbeat failed: {e}")
                return {}

    async def heartbeat_loop(self):
        while True:
            await self.send_heartbeat()
            await asyncio.sleep(self.heartbeat_interval)

    async def start(self, port: int):
        import uvicorn

        config = uvicorn.Config(
            self.app,
            host="0.0.0.0",
            port=port,
            log_level="info",
        )
        server = uvicorn.Server(config)
        await asyncio.gather(server.serve(), self.heartbeat_loop())