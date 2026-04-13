from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    GROQ_API_KEY:str
    GROQ_MODEL:str = 'llama-3.3-70b-versatile'


settings = Settings()