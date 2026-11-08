from pydantic_settings import BaseSettings, SettingsConfigDict

from functools import lru_cache


class Settings(BaseSettings):
    database_url: str

    # use whatever model you like

    openai_model: str
    openai_api_key: str

    embedding_model: str
    embedding_filename: str = "onnx/model_qint8_arm64.onnx"
    
    reranker_model: str
    reranker_filename: str = "onnx/model_qint8_arm64.onnx"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()