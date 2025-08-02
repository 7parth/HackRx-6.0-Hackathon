from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

# Load environment variables
env_path = os.path.join("App", ".env")
load_dotenv(dotenv_path=env_path)

class Settings(BaseSettings):
    GEMINI_API_KEY: str
    # Optional configuration with defaults
    DEFAULT_CHUNK_SIZE: int = 1000
    DEFAULT_CHUNK_OVERLAP: int = 300
    DEFAULT_RETRIEVER_K: int = 8
    MAX_DOCUMENT_SIZE_MB: int = 1024  # Increased from 50MB to 1GB or higher
    CACHE_SIZE_LIMIT: int = 100

    class Config:
        env_file = env_path
        env_file_encoding = 'utf-8'

    def validate_api_key(self) -> bool:
        """Validate that the API key is present and not empty"""
        return bool(self.GEMINI_API_KEY and self.GEMINI_API_KEY.strip())

# Singleton pattern with validation
try:
    settings = Settings()
    if not settings.validate_api_key():
        raise ValueError("GEMINI_API_KEY is required but not found in environment variables")
except Exception as e:
    print(f"Configuration error: {e}")
    raise
