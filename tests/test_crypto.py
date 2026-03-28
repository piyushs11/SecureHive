import sys, os, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.crypto.utils import aes_encrypt, aes_decrypt, generate_keypair, sign_payload, verify_signature, load_private_key, load_public_key

KEY = bytes.fromhex("a" * 64)   # 32-byte test key

def test_aes_roundtrip():
    msg = b"hello securehive"
    enc = aes_encrypt(msg, KEY)
    assert aes_decrypt(enc, KEY) == msg

def test_aes_tamper_detection():
    enc = aes_encrypt(b"secret", KEY)
    enc["ciphertext"] = enc["ciphertext"][:-4] + "ZZZZ"
    with pytest.raises(ValueError, match="tampered"):
        aes_decrypt(enc, KEY)

def test_aes_nonce_uniqueness():
    e1 = aes_encrypt(b"same", KEY)
    e2 = aes_encrypt(b"same", KEY)
    assert e1["nonce"] != e2["nonce"]   # fresh nonce every call

def test_ed25519_roundtrip(tmp_path):
    generate_keypair("test_agent", keys_dir=str(tmp_path))
    priv = load_private_key(f"{tmp_path}/test_agent_private.pem")
    pub  = load_public_key(f"{tmp_path}/test_agent_public.pem")
    payload = b"telemetry data"
    sig = sign_payload(payload, priv)
    assert verify_signature(payload, sig, pub) is True

def test_ed25519_tamper():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        generate_keypair("t", keys_dir=tmp)
        priv = load_private_key(f"{tmp}/t_private.pem")
        pub  = load_public_key(f"{tmp}/t_public.pem")
        sig = sign_payload(b"real data", priv)
        assert verify_signature(b"tampered data", sig, pub) is False