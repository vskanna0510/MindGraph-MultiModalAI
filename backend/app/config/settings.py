"""Application configuration loaded from environment variables and YAML files."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_yaml_config(filename: str) -> dict[str, Any]:
    """Load a YAML configuration file from the configs directory."""
    config_path = _project_root() / "configs" / filename
    if not config_path.exists():
        return {}
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


class Settings(BaseSettings):
    """Application settings with environment variable overrides."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "MindGraph++"
    app_env: str = "development"
    app_debug: bool = False
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    api_version: str = "v1"

    # Security
    secret_key: str = Field(min_length=16)
    encryption_key: str = Field(min_length=16)
    jwt_secret: str = Field(min_length=32)
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7

    # PostgreSQL
    database_url: str
    postgres_user: str = "mindgraph"
    postgres_password: str = "mindgraph"
    postgres_db: str = "mindgraph"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "mindgraph_neo4j"
    neo4j_database: str = "neo4j"

    # Celery
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # Directories
    model_directory: str = "./ml_pipeline/weights"
    dataset_directory: str = "./datasets"
    upload_directory: str = "./assets/uploads"
    log_directory: str = "./backend/logs"
    config_directory: str = "./configs"

    # Object storage
    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "mindgraph-media"
    s3_region: str = "us-east-1"
    s3_use_ssl: bool = False

    # Differential privacy
    dp_enabled: bool = False
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5

    # SDN
    sdn_enabled: bool = False
    sdn_controller_host: str = "localhost"
    sdn_controller_port: int = 6633

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    # CORS
    cors_origins: str = "http://localhost:3000,http://localhost:8080"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | list[str]) -> str:
        if isinstance(value, list):
            return ",".join(value)
        return value

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def is_development(self) -> bool:
        return self.app_env.lower() == "development"

    def get_yaml(self, filename: str) -> dict[str, Any]:
        return load_yaml_config(filename)


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
