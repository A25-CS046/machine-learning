import os
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DatabaseConfig:
    url: str = field(default_factory=lambda: os.getenv('DATABASE_URL', 'sqlite:///pm_app.db'))
    pool_size: int = field(default_factory=lambda: int(os.getenv('DB_POOL_SIZE', '10')))
    max_overflow: int = field(default_factory=lambda: int(os.getenv('DB_MAX_OVERFLOW', '20')))
    echo: bool = field(default_factory=lambda: os.getenv('DB_ECHO', 'false').lower() == 'true')


@dataclass
class ModelStorageConfig:
    backend: Literal['local', 'gcs'] = field(default_factory=lambda: os.getenv('MODEL_STORAGE_BACKEND', 'local'))
    local_path: str = field(default_factory=lambda: os.getenv('MODEL_STORAGE_PATH', './artifacts'))
    gcs_bucket: str | None = field(default_factory=lambda: os.getenv('GCS_MODEL_BUCKET'))


@dataclass
class GeminiConfig:
    api_key: str | None = field(default_factory=lambda: os.getenv('GEMINI_API_KEY'))
    model_name: str = field(default_factory=lambda: os.getenv('GEMINI_MODEL_NAME', 'gemini-2.0-flash-exp'))
    api_url: str = field(default_factory=lambda: os.getenv('GEMINI_API_URL', 'https://generativelanguage.googleapis.com/v1beta'))
    timeout: int = field(default_factory=lambda: int(os.getenv('GEMINI_TIMEOUT', '60')))


@dataclass
class AppConfig:
    flask_env: str = field(default_factory=lambda: os.getenv('FLASK_ENV', 'development'))
    secret_key: str = field(default_factory=lambda: os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production'))
    log_level: str = field(default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO'))
    log_format: Literal['text', 'json'] = field(default_factory=lambda: os.getenv('LOG_FORMAT', 'text'))
    cors_origins: list[str] = field(default_factory=lambda: os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://localhost:5173').split(','))
    
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    model_storage: ModelStorageConfig = field(default_factory=ModelStorageConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    
    @property
    def is_production(self) -> bool:
        return self.flask_env == 'production'
    
    @property
    def is_development(self) -> bool:
        return self.flask_env == 'development'


def load_config() -> AppConfig:
    return AppConfig()
