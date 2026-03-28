import json
import os

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.crypto.utils import aes_decrypt, load_public_key, verify_signature
from src.coordinator.trust_engine import TrustEngine
from src.detection.semantic_divergence import SemanticDivergenceDetector

def _get_aes_key() -> bytes:
    key_hex = os.environ.get("SHARED_AES_KEY", "a" * 64)
    return bytes.fromhex(key_hex)

def _get_keys_dir() -> str:
    return os.environ.get("KEYS_DIR", "./keys")

app = FastAPI(title="PrivAgent Coordinator", version="1.0.0")

trust_engine = TrustEngine(
    alpha=0.9,
    quarantine_threshold=0.4,
    suspicious_threshold=0.7,
    normalization_ceiling=100.0,   
)
detector = SemanticDivergenceDetector(
    window_size=50,
    alert_threshold=None,    
    warmup_samples=20,
    alert_z_score=3.0,
)

class TelemetryEnvelope(BaseModel):
    agent_id: str
    envelope: dict   # {"nonce": "...", "ciphertext": "..."}

@app.get("/health")
def health():
    return {"status": "ok", "service": "coordinator"}

@app.post("/telemetry")
async def receive_telemetry(msg: TelemetryEnvelope):
    aes_key  = _get_aes_key()
    keys_dir = _get_keys_dir()

    try:
        decrypted_bytes = aes_decrypt(msg.envelope, aes_key)
        inner = json.loads(decrypted_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Malformed envelope: {e}")

    payload: dict  = inner.get("payload", {})
    signature: str = inner.get("signature", "")
    agent_id: str  = payload.get("agent_id", "")

    if not agent_id:
        raise HTTPException(status_code=400, detail="Missing agent_id in payload")

    if msg.agent_id != agent_id:
        raise HTTPException(
            status_code=403,
            detail=f"Identity mismatch: envelope claims '{msg.agent_id}' but payload is signed by '{agent_id}'"
        )

    try:
        pub_key = load_public_key(f"{keys_dir}/{agent_id}_public.pem")
    except FileNotFoundError:
        raise HTTPException(status_code=403, detail=f"Unknown agent: {agent_id}")

    payload_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
    if not verify_signature(payload_bytes, signature, pub_key):
        raise HTTPException(
            status_code=403, detail=f"Invalid signature from {agent_id}"
        )

    dp_embedding = np.array(payload.get("dp_embedding", []), dtype=float)
    if dp_embedding.shape[0] == 0:
        raise HTTPException(status_code=400, detail="Missing dp_embedding")

    detection = detector.score_embedding(agent_id, dp_embedding)

    record = trust_engine.update(agent_id, divergence=detection["divergence"])

    status = record.status(
        quarantine_t=trust_engine.quarantine_threshold,
        suspicious_t=trust_engine.suspicious_threshold,
    )
    print(
        f"[coordinator] {agent_id:20s} | "
        f"div={detection['divergence']:6.3f} | "
        f"trust={record.score:.4f} | {status}"
    )

    return {
        "status": "ok",
        "agent_id": agent_id,
        "trust_score": record.score,
        "agent_status": status,
        "divergence": detection["divergence"],
        "alert": detection["alert"],
        "heartbeat_count": record.heartbeat_count,
    }

@app.get("/trust")
def get_all_trust_scores():
    return trust_engine.get_all_scores()


@app.get("/trust/{agent_id}")
def get_agent_trust(agent_id: str):
    record = trust_engine.get_record(agent_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No record for {agent_id}")
    return {
        "agent_id": record.agent_id,
        "score": record.score,
        "status": record.status(
            quarantine_t=trust_engine.quarantine_threshold,
            suspicious_t=trust_engine.suspicious_threshold,
        ),
        "heartbeat_count": record.heartbeat_count,
        "last_seen": record.last_seen,
        "history": record.history[-20:],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)