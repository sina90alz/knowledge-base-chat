"""Application configuration settings."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings:
    """Application settings from environment variables."""

    # Application
    APP_NAME: str = os.getenv("APP_NAME", "knowledge-base-chat")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # API
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # Model
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    VECTOR_STORE_PATH: Path = Path(os.getenv("VECTOR_STORE_PATH", "data/vector_store"))

    def __init__(self):
        """Initialize settings and ensure directories exist."""
        self.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
