import os

os.environ["SHARED_AES_KEY"] = "a" * 64
os.environ["COORDINATOR_URL"] = "http://localhost:8000"
os.environ["OLLAMA_URL"]      = "http://localhost:11434"
os.environ["KEYS_DIR"]        = "./keys"