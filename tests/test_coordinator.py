import sys, os, json, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("SHARED_AES_KEY", "a" * 64)

from fastapi.testclient import TestClient
from src.crypto.utils import (
    aes_encrypt, generate_keypair,
    load_private_key, sign_payload,
)
import numpy as np

KEY = bytes.fromhex("a" * 64)


def _make_valid_envelope(agent_id: str, keys_dir: str) -> dict:
    generate_keypair(agent_id, keys_dir=keys_dir)
    priv = load_private_key(f"{keys_dir}/{agent_id}_private.pem")
    payload = {
        "agent_id": agent_id,
        "timestamp": 1000.0,
        "epsilon_used": 1.0,
        "dp_embedding": np.random.randn(384).tolist(),
        "action_count": 5,
    }
    payload_bytes = json.dumps(payload, sort_keys=True).encode()
    sig = sign_payload(payload_bytes, priv)
    inner = json.dumps({"payload": payload, "signature": sig}).encode()
    envelope = aes_encrypt(inner, KEY)
    return envelope


def test_health_endpoint():
    from src.coordinator.main import app
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_telemetry_accepted(tmp_path):
    os.environ["KEYS_DIR"] = str(tmp_path)
    from src.coordinator.main import app
    client = TestClient(app)
    envelope = _make_valid_envelope("planner", str(tmp_path))
    resp = client.post(
        "/telemetry",
        json={"agent_id": "planner", "envelope": envelope},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["agent_id"] == "planner"
    assert 0.0 <= data["trust_score"] <= 1.0


def test_tampered_packet_rejected(tmp_path):
    os.environ["KEYS_DIR"] = str(tmp_path)
    from src.coordinator.main import app
    client = TestClient(app)
    envelope = _make_valid_envelope("planner", str(tmp_path))
    envelope["ciphertext"] = envelope["ciphertext"][:-4] + "ZZZZ"
    resp = client.post(
        "/telemetry",
        json={"agent_id": "planner", "envelope": envelope},
    )
    assert resp.status_code == 400


def test_unknown_agent_rejected(tmp_path):
    os.environ["KEYS_DIR"] = str(tmp_path)
    from src.coordinator.main import app
    client = TestClient(app)
    envelope = _make_valid_envelope("agent_x", str(tmp_path))
    resp = client.post(
        "/telemetry",
        json={"agent_id": "unknown_agent", "envelope": envelope},
    )
    assert resp.status_code == 403


def test_trust_endpoint_empty():
    from src.coordinator.main import app
    client = TestClient(app)
    resp = client.get("/trust")
    assert resp.status_code == 200
    assert isinstance(resp.json(), dict)