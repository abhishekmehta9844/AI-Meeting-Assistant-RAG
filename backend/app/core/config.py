from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Meeting Assistant"
    # Fallback to sqlite if postgres is not configured
    DATABASE_URL: str = "sqlite:///./meetingdb.sqlite3"
    GROQ_API_KEY: str = ""
    PYANNOTE_API_KEY: str = ""
    SARVAM_API_KEY: str = ""

    model_config = {
        "env_file": ".env",
        "extra": "ignore"
    }

settings = Settings()

