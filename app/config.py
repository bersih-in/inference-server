from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Bersihin Server Inference"

    # backend endpoint for inference
    BACKEND_ENDPOINT: str

    model_config = SettingsConfigDict(env_file=".env")