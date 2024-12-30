from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_storage_dir: str
    default_model_social: str
    default_model_news: str
    model_meta_file: str
    logs_dir: str
    num_cores: int

    class Config:
        env_file = ".env"
