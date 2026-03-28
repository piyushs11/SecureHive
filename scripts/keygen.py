import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.crypto.utils import generate_keypair

AGENTS = ["coordinator", "planner", "retriever", "policy_checker", "executor"]

if __name__ == "__main__":
    keys_dir = "./keys"
    for name in AGENTS:
        generate_keypair(name, keys_dir=keys_dir)
    print(f"\n[keygen] Done. {len(AGENTS) * 2} PEM files in {keys_dir}/")
    print("[keygen] These are in .gitignore — share via secure channel with your partner.")