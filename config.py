from dataclasses import dataclass
import os

@dataclass
class Settings:
    provider: str = os.getenv("PROVIDER", "gemini")
    # Gemini
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "AIzaSyCeaDA9dKx9jARJJnsG2PyM71ajiNuGIj0")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")

    # Embeddings
    text_model_name: str = os.getenv("TEXT_EMBEDDER", "sentence-transformers/all-MiniLM-L6-v2")
    clip_model_name: str = os.getenv("CLIP_EMBEDDER", "clip-ViT-B-32")

    # Stores
    chroma_dir: str = os.getenv("CHROMA_DIR", "./chroma_db")
    sqlite_path: str = os.getenv("SQLITE_PATH", "./facts.db")

    # RAG
    top_k: int = int(os.getenv("TOP_K", 5))
    max_tokens: int = int(os.getenv("MAX_TOKENS", 600))
    
    # Logging
    environment: str = os.getenv("ENVIRONMENT", "development")
    dev_log_level: str = os.getenv("DEV_LOG_LEVEL", "DEBUG")
    prod_log_level: str = os.getenv("PROD_LOG_LEVEL", "INFO")
    log_dir: str = os.getenv("LOG_DIR", "logs")

settings = Settings()
