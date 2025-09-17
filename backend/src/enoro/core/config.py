"""
Core configuration and settings for Enoro.
"""

import os
from typing import List
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define application paths
HOME_DIR = Path.home()
ENORO_DATA_DIR = HOME_DIR / ".enoro"
ENORO_DB_PATH = ENORO_DATA_DIR / "data" / "enoro.db"
ENORO_LOGS_DIR = ENORO_DATA_DIR / "data" / "logs"
ENORO_CACHE_DIR = ENORO_DATA_DIR / "data" / "cache"
ENORO_CONFIG_DIR = ENORO_DATA_DIR / "config"


@dataclass
class Settings:
    """Application settings."""

    # Application
    app_name: str = "Enoro Video Discovery"
    app_version: str = "1.0.0"
    debug: bool = False

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"

    # Database
    database_url: str = f"sqlite:///{ENORO_DB_PATH}"

    # Application data directories
    data_dir: str = str(ENORO_DATA_DIR)
    logs_dir: str = str(ENORO_LOGS_DIR)
    cache_dir: str = str(ENORO_CACHE_DIR)
    config_dir: str = str(ENORO_CONFIG_DIR)

    # YouTube API
    youtube_api_key: str = ""
    youtube_quota_limit: int = 10000

    # YouTube OAuth
    google_client_id: str = ""
    google_client_secret: str = ""
    google_redirect_uri: str = "http://localhost:8000/api/v1/auth/youtube/callback"
    youtube_scopes: List[str] = None  # type: ignore

    # ML Settings
    ml_min_ratings: int = 10
    ml_model_retrain_threshold: int = 5

    # Redis Cache (optional)
    redis_url: str = ""
    cache_ttl: int = 3600  # 1 hour

    # CORS
    cors_origins: List[str] = None  # type: ignore

    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds

    def __post_init__(self):
        """Load settings from environment variables and ensure directories exist."""
        # Ensure ~/.enoro directories exist
        ENORO_DATA_DIR.mkdir(parents=True, exist_ok=True)
        ENORO_LOGS_DIR.mkdir(parents=True, exist_ok=True)
        ENORO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (ENORO_CACHE_DIR / "channels").mkdir(exist_ok=True)
        (ENORO_CACHE_DIR / "videos").mkdir(exist_ok=True)
        ENORO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        if self.cors_origins is None:
            self.cors_origins = ["http://localhost:3000", "http://localhost:3001"]

        if self.youtube_scopes is None:
            self.youtube_scopes = [
                "https://www.googleapis.com/auth/youtube.readonly",
                "https://www.googleapis.com/auth/youtube.force-ssl",
            ]

        # Override with environment variables
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.api_host = os.getenv("API_HOST", self.api_host)
        self.api_port = int(os.getenv("API_PORT", self.api_port))

        # Use production database path, but allow override
        env_db_url = os.getenv("DATABASE_URL")
        if env_db_url:
            self.database_url = env_db_url

        self.youtube_api_key = os.getenv("YOUTUBE_API_KEY", self.youtube_api_key)
        self.youtube_quota_limit = int(
            os.getenv("YOUTUBE_QUOTA_LIMIT", self.youtube_quota_limit)
        )
        self.google_client_id = os.getenv("GOOGLE_CLIENT_ID", self.google_client_id)
        self.google_client_secret = os.getenv(
            "GOOGLE_CLIENT_SECRET", self.google_client_secret
        )
        self.google_redirect_uri = os.getenv(
            "GOOGLE_REDIRECT_URI", self.google_redirect_uri
        )
        self.ml_min_ratings = int(os.getenv("ML_MIN_RATINGS", self.ml_min_ratings))
        self.ml_model_retrain_threshold = int(
            os.getenv("ML_RETRAIN_THRESHOLD", self.ml_model_retrain_threshold)
        )
        self.redis_url = os.getenv("REDIS_URL", self.redis_url)
        self.cache_ttl = int(os.getenv("CACHE_TTL", self.cache_ttl))
        self.rate_limit_requests = int(
            os.getenv("RATE_LIMIT_REQUESTS", self.rate_limit_requests)
        )
        self.rate_limit_window = int(
            os.getenv("RATE_LIMIT_WINDOW", self.rate_limit_window)
        )


# Global settings instance
settings = Settings()
