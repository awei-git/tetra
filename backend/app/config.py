from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any
import os
import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    # App settings
    app_name: str = "Tetra WebGUI Backend"
    debug: bool = False
    
    # Database
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:postgres@localhost/tetra"
    )
    
    # CORS
    cors_origins: list[str] = [
        "http://localhost:5173",  # Vue dev server
        "http://localhost:5174",  # Vue preview
        "http://localhost:3000",  # Alternative frontend
        "http://localhost:5175",  # Additional Vite ports
        "http://localhost:5176",
        "http://localhost:5177",
    ]
    
    # LLM settings - will be overridden by secrets.yml
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    llm_provider: str = "openai"  # anthropic or openai
    llm_model: str = "gpt-4"  # or claude-3-sonnet-20240229
    
    # Security
    secret_key: str = os.getenv("SECRET_KEY", "your-secret-key-here-change-in-production")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # WebSocket
    ws_heartbeat_interval: int = 30  # seconds
    
    def __init__(self, **data):
        super().__init__(**data)
        self._load_secrets()
    
    def _load_secrets(self) -> None:
        """Load secrets from the secrets.yml file"""
        # Try multiple paths
        possible_paths = [
            Path("../config/secrets.yml"),  # From backend directory
            Path("config/secrets.yml"),     # From root
            Path("../../config/secrets.yml"), # Alternative
        ]
        
        for secrets_path in possible_paths:
            if secrets_path.exists():
                logger.info(f"Loading secrets from: {secrets_path}")
                with open(secrets_path, 'r') as f:
                    secrets = yaml.safe_load(f)
                    
                # Override with values from secrets.yml
                if 'api_keys' in secrets:
                    if 'openai' in secrets['api_keys']:
                        if isinstance(secrets['api_keys']['openai'], dict):
                            self.openai_api_key = secrets['api_keys']['openai'].get('api_key')
                        else:
                            self.openai_api_key = secrets['api_keys']['openai']
                        logger.info("OpenAI API key loaded from secrets.yml")
                    if 'anthropic' in secrets['api_keys']:
                        self.anthropic_api_key = secrets['api_keys'].get('anthropic')
                    
                    # Use OpenAI since we have the key
                    if self.openai_api_key:
                        self.llm_provider = "openai"
                        self.llm_model = "gpt-4"
                        logger.info(f"Using OpenAI with model: {self.llm_model}")
                        
                if 'security' in secrets:
                    self.secret_key = secrets['security'].get('secret_key', self.secret_key)
                
                break
        else:
            logger.warning("No secrets.yml file found")
    
    class Config:
        env_file = ".env"


# Create global settings instance
settings = Settings()