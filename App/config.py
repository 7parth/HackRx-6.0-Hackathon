from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

env_path = os.path.join("App", ".env")
load_dotenv(dotenv_path=env_path)

class Settings(BaseSettings):
    GEMINI_API_KEY: str

    class Config:
        env_file = env_path

settings = Settings()  # Singleton pattern
