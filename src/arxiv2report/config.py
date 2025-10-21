from pydantic_settings import BaseSettings
from pydantic import ConfigDict

class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env")

    openai_api_key: str
    openai_api_base: str = "https://www.chataiapi.com"
    netmind_api_key: str = ""

settings = Settings()
