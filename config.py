from pydantic import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4"
    MAX_NEWS_RESULTS: int = 5
    
    class Config:
        env_file = ".env"

settings = Settings() 