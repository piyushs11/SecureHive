import os
import base64
import json
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey
)
from cryptography.hazmat.primitives.serialization import (
    load_pem_private_key, load_pem_public_key,
    Encoding, PrivateFormat, PublicFormat, NoEncryption
)
from cryptography.exceptions import InvalidTag

def aes_encrypt(plaintext: bytes, key: bytes) -> dict:
    if len(key) != 32:
        raise ValueError(f"AES-256 needs 32-byte key, got {len(key)}")
    nonce = os.urandom(12)  # 96-bit nonce — GCM standard
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, associated_data=None)
    return {
        "nonce": base64.b64encode(nonce).decode("utf-8"),
        "ciphertext": base64.b64encode(ciphertext).decode("utf-8"),
    }

def aes_decrypt(encrypted: dict, key: bytes) -> bytes:
    try:
        nonce = base64.b64decode(encrypted["nonce"])
        ciphertext = base64.b64decode(encrypted["ciphertext"])
        return AESGCM(key).decrypt(nonce, ciphertext, associated_data=None)
    except InvalidTag:
        raise ValueError("Decryption failed — message was tampered with or key mismatch")
    except KeyError as e:
        raise ValueError(f"Malformed envelope — missing field: {e}")

def generate_keypair(agent_id: str, keys_dir: str = "./keys") -> None:
    os.makedirs(keys_dir, exist_ok=True)
    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key()
    priv_bytes = priv.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
    pub_bytes = pub.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    with open(f"{keys_dir}/{agent_id}_private.pem", "wb") as f:
        f.write(priv_bytes)
    with open(f"{keys_dir}/{agent_id}_public.pem", "wb") as f:
        f.write(pub_bytes)
    print(f"[keygen] Keys written for '{agent_id}'")

def load_private_key(path: str) -> Ed25519PrivateKey:
    with open(path, "rb") as f:
        return load_pem_private_key(f.read(), password=None)

def load_public_key(path: str) -> Ed25519PublicKey:
    with open(path, "rb") as f:
        return load_pem_public_key(f.read())

def sign_payload(payload_bytes: bytes, private_key: Ed25519PrivateKey) -> str:
    return base64.b64encode(private_key.sign(payload_bytes)).decode("utf-8")

def verify_signature(
    payload_bytes: bytes,
    signature_b64: str,
    public_key: Ed25519PublicKey
) -> bool:
    try:
        sig = base64.b64decode(signature_b64)
        public_key.verify(sig, payload_bytes)
        return True
    except Exception:
        return False