from pydantic_settings import BaseSettings
from dotenv import load_dotenv


# Define the absolute path to the .env file
env_path = r"App\.env"

# Explicitly load the .env file
load_dotenv(dotenv_path=env_path)

class Settings(BaseSettings):
    GEMINI_API_KEY: str

    class Config:
        env_file = env_path  # Use the same absolute path here

settings = Settings() # type: ignore

