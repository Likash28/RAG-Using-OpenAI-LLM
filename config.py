from dataclasses import dataclass
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class Settings:
    provider: str = os.getenv("PROVIDER", "openai")
    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Embeddings
    text_model_name: str = os.getenv("TEXT_EMBEDDER", "sentence-transformers/all-MiniLM-L6-v2")
    blip_model_name: str = os.getenv("BLIP_EMBEDDER", "Salesforce/blip-itm-base-coco")

    # Stores
    chroma_dir: str = os.getenv("CHROMA_DIR", "./chroma_db")
    sqlite_path: str = os.getenv("SQLITE_PATH", "./facts.db")

    # RAG
    top_k: int = int(os.getenv("TOP_K", 5))
    max_tokens: int = int(os.getenv("MAX_TOKENS", 600))
    
    # Audio transcription
    openai_whisper: str = os.getenv("OPENAI_WHISPER", "local")  # local|none (uses faster-whisper)
    whisper_model_size: str = os.getenv("WHISPER_MODEL_SIZE", "base")  # tiny|base|small|medium|large-v2|large-v3
    
    # Logging
    environment: str = os.getenv("ENVIRONMENT", "development")
    dev_log_level: str = os.getenv("DEV_LOG_LEVEL", "DEBUG")
    prod_log_level: str = os.getenv("PROD_LOG_LEVEL", "INFO")
    log_dir: str = os.getenv("LOG_DIR", "logs")

settings = Settings()

# Validate required settings only when needed
def validate_settings():
    if settings.provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required when PROVIDER=openai")
    else:
        raise ValueError(f"Unknown provider: {settings.provider}. Use 'openai'")