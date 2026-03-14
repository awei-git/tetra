import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application
    app_name: str = "tetra"
    app_env: str = "development"
    log_level: str = "INFO"
    
    # Database
    database_host: str = "localhost"
    database_port: int = 5432
    database_name: str = "tetra"
    database_user: str = "tetra_user"
    
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_market_data: str = "market-data"
    kafka_topic_events: str = "events"
    kafka_topic_alerts: str = "alerts"
    kafka_consumer_group: str = "tetra-consumer-group"
    
    # TimescaleDB settings
    timescale_chunk_interval: str = "7 days"
    timescale_compression_after: str = "30 days"
    timescale_retention_period: str = "2 years"
    
    # Data ingestion
    batch_size: int = 1000
    retry_attempts: int = 3
    retry_delay: int = 5  # seconds
    
    # Rate limits (calls per minute)
    polygon_rate_limit: int = 5000  # Basic tier
    finnhub_rate_limit: int = 60
    fred_rate_limit: int = 120
    news_rate_limit: int = 500
    
    # API URLs
    polygon_base_url: str = "https://api.polygon.io"
    finnhub_base_url: str = "https://finnhub.io/api/v1"
    fred_base_url: str = "https://api.stlouisfed.org/fred"
    news_api_base_url: str = "https://newsapi.org/v2"
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    polymarket_gamma_base_url: str = "https://gamma-api.polymarket.com"
    
    # LLM models
    openai_model: str = "gpt-5.2"
    deepseek_model: str = "deepseek-chat"
    gemini_model: str = "gemini-3.1-pro-preview"
    claude_model: str = "claude-opus-4-6"

    # Email / report delivery
    email_enabled: bool = True
    email_smtp_server: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    # email_recipients: loaded from secrets.yml via property

    # Features
    enable_paper_trading: bool = True
    enable_live_trading: bool = False
    enable_ml_predictions: bool = True
    enable_llm_analysis: bool = True
    enable_email_reports: bool = True
    
    # Monitoring
    prometheus_port: int = 9090
    grafana_port: int = 3000
    
    # Storage schemas
    storage_schemas: list = ["market_data", "event", "derived", "strategies", "execution"]
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    def __init__(self, **data):
        super().__init__(**data)
        self._secrets = self._load_secrets()
    
    def _load_secrets(self) -> Dict[str, Any]:
        """Load secrets from the secrets.yml file"""
        candidates = [
            Path("config/secrets.yml"),
            Path(__file__).resolve().parent / "secrets.yml",
            Path(__file__).resolve().parents[1] / "config" / "secrets.yml",
        ]

        for secrets_path in candidates:
            if secrets_path.exists():
                with open(secrets_path, "r") as f:
                    return yaml.safe_load(f) or {}

        # Return empty dict if no secrets file found
        return {}
    
    @property
    def database_password(self) -> str:
        return self._secrets.get("database", {}).get("password", "tetra_password")
    
    @property
    def database_url(self) -> str:
        return f"postgresql+asyncpg://{self.database_user}:{self.database_password}@{self.database_host}:{self.database_port}/{self.database_name}"
    
    @property
    def sync_database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.database_user}:{self.database_password}"
            f"@{self.database_host}:{self.database_port}/{self.database_name}"
        )
    
    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    # API Keys properties
    @property
    def polygon_api_key(self) -> Optional[str]:
        return self._secrets.get("api_keys", {}).get("polygon")
    
    @property
    def finnhub_api_key(self) -> Optional[str]:
        return self._secrets.get("api_keys", {}).get("finnhub")
    
    @property
    def news_api_key(self) -> Optional[str]:
        return self._secrets.get("api_keys", {}).get("news_api")
    
    @property
    def fred_api_key(self) -> Optional[str]:
        return self._secrets.get("api_keys", {}).get("fred")
    
    @property
    def alphavantage_api_key(self) -> Optional[str]:
        return self._secrets.get("api_keys", {}).get("alphavantage")

    @property
    def sec_user_agent(self) -> Optional[str]:
        return self._secrets.get("api_keys", {}).get("sec_user_agent")
    
    @property
    def openai_api_key(self) -> Optional[str]:
        return self._secrets.get("api_keys", {}).get("openai", {}).get("api_key")
    
    @property
    def openai_org(self) -> Optional[str]:
        return self._secrets.get("api_keys", {}).get("openai", {}).get("organization")
    
    @property
    def deepseek_api_key(self) -> Optional[str]:
        return self._secrets.get("api_keys", {}).get("deepseek")

    @property
    def gemini_api_key(self) -> Optional[str]:
        return self._secrets.get("api_keys", {}).get("gemini")

    @property
    def gemini_models(self) -> List[str]:
        models = self._secrets.get("api_keys", {}).get("gemini_models")
        if models is None:
            models = self._secrets.get("llm", {}).get("gemini_models")
        if isinstance(models, str):
            return [item.strip() for item in models.split(",") if item.strip()]
        if isinstance(models, list):
            return [str(item).strip() for item in models if str(item).strip()]
        return []
    
    @property
    def azure_speech_key(self) -> Optional[str]:
        return self._secrets.get("api_keys", {}).get("azure_speech")

    @property
    def anthropic_api_key(self) -> Optional[str]:
        return self._secrets.get("api_keys", {}).get("anthropic")

    @property
    def smtp_username(self) -> Optional[str]:
        return self._secrets.get("email", {}).get("smtp_username")

    @property
    def smtp_password(self) -> Optional[str]:
        return self._secrets.get("email", {}).get("smtp_password")

    @property
    def email_recipients(self) -> str:
        return self._secrets.get("email", {}).get("recipients", "")

    @property
    def portfolio_config(self) -> dict:
        return self._secrets.get("Portfolio", {})

    @property
    def polymarket_api_key(self) -> Optional[str]:
        return self._secrets.get("api_keys", {}).get("polymarket", {}).get("api_key")

    @property
    def polymarket_secret(self) -> Optional[str]:
        return self._secrets.get("api_keys", {}).get("polymarket", {}).get("secret")

    @property
    def polymarket_passphrase(self) -> Optional[str]:
        return self._secrets.get("api_keys", {}).get("polymarket", {}).get("passphrase")
    
    @property
    def secret_key(self) -> str:
        return self._secrets.get("security", {}).get("secret_key", "default-secret-key")
    
    @property
    def api_key_salt(self) -> str:
        return self._secrets.get("security", {}).get("api_key_salt", "default-salt")


# Global settings instance
settings = Settings()
