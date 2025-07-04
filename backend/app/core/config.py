from pydantic_settings import BaseSettings
from pydantic import validator
from typing import List, Optional
import os

class Settings(BaseSettings):
    """Application settings from environment variables."""
    
    # Application
    APP_NAME: str = "Terminal++"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    
    # Security
    SECRET_KEY: str
    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 1440  # 24 hours
    ENCRYPTION_KEY: str
    
    # Database
    DATABASE_URL: str
    DATABASE_ECHO: bool = False
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    
    # AI Services
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GOOGLE_AI_API_KEY: Optional[str] = None
    DEFAULT_AI_MODEL: str = "gpt-4"
    
    # Container Settings
    DOCKER_HOST: str = "unix:///var/run/docker.sock"
    CONTAINER_REGISTRY: str = "terminal-plus-plus"
    MAX_CONTAINERS_PER_USER: int = 5
    CONTAINER_TIMEOUT_MINUTES: int = 60
    CONTAINER_AUTO_CLEANUP: bool = True
    
    # Network & Security
    ALLOWED_HOSTS: List[str] = ["*"]
    ALLOWED_ORIGINS: List[str] = ["*"]
    CORS_CREDENTIALS: bool = True
    
    # File Storage
    STORAGE_BACKEND: str = "local"  # local, s3, gcs
    STORAGE_PATH: str = "/app/storage"
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_BUCKET_NAME: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    LOG_LEVEL: str = "INFO"
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # seconds
    
    # Email (for notifications)
    SMTP_SERVER: Optional[str] = None
    SMTP_PORT: int = 587
    SMTP_USERNAME: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    FROM_EMAIL: Optional[str] = None
    
    # WebSocket
    WS_HEARTBEAT_INTERVAL: int = 30
    WS_MAX_CONNECTIONS: int = 1000
    
    # Security Policies
    ENABLE_THREAT_DETECTION: bool = True
    MAX_CPU_PERCENT: int = 80
    MAX_MEMORY_MB: int = 1024
    MAX_DISK_MB: int = 2048
    MAX_PROCESSES: int = 100
    BLOCKED_COMMANDS: List[str] = [
        "rm -rf /",
        "dd if=/dev/zero",
        ":(){ :|:& };:",
        "sudo su -"
    ]
    
    # Backup & Recovery
    BACKUP_ENABLED: bool = True
    BACKUP_INTERVAL_HOURS: int = 24
    BACKUP_RETENTION_DAYS: int = 30
    
    @validator("ALLOWED_ORIGINS", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("ALLOWED_HOSTS", pre=True)
    def parse_allowed_hosts(cls, v):
        if isinstance(v, str):
            return [host.strip() for host in v.split(",")]
        return v
    
    @validator("BLOCKED_COMMANDS", pre=True)
    def parse_blocked_commands(cls, v):
        if isinstance(v, str):
            return [cmd.strip() for cmd in v.split(",")]
        return v
    
    @validator("DATABASE_URL", pre=True)
    def build_database_url(cls, v):
        if v and not v.startswith(("postgresql://", "sqlite://", "mysql://")):
            # Build from components if not a full URL
            return f"postgresql://{v}"
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()

# Database configuration
class DatabaseConfig:
    """Database connection configuration."""
    
    @staticmethod
    def get_database_url() -> str:
        return settings.DATABASE_URL
    
    @staticmethod
    def get_pool_settings() -> dict:
        return {
            "pool_size": 20,
            "max_overflow": 40,
            "pool_pre_ping": True,
            "pool_recycle": 3600,
            "echo": settings.DATABASE_ECHO
        }

# Redis configuration
class RedisConfig:
    """Redis connection configuration."""
    
    @staticmethod
    def get_redis_url() -> str:
        return settings.REDIS_URL
    
    @staticmethod
    def get_redis_settings() -> dict:
        return {
            "decode_responses": True,
            "max_connections": 100,
            "retry_on_timeout": True,
            "socket_keepalive": True,
            "socket_keepalive_options": {}
        }

# Container configuration
class ContainerConfig:
    """Container orchestration configuration."""
    
    @staticmethod
    def get_security_policy() -> dict:
        return {
            "max_cpu_percent": settings.MAX_CPU_PERCENT,
            "max_memory_mb": settings.MAX_MEMORY_MB,
            "max_disk_mb": settings.MAX_DISK_MB,
            "max_processes": settings.MAX_PROCESSES,
            "blocked_commands": settings.BLOCKED_COMMANDS,
            "network_access": True,
            "internet_access": True
        }
    
    @staticmethod
    def get_resource_limits() -> dict:
        return {
            "cpu_limit": "1000m",  # 1 CPU core
            "memory_limit": f"{settings.MAX_MEMORY_MB}Mi",
            "disk_limit": f"{settings.MAX_DISK_MB}Mi",
            "pids_limit": settings.MAX_PROCESSES
        }

# AI configuration
class AIConfig:
    """AI service configuration."""
    
    @staticmethod
    def get_openai_config() -> dict:
        return {
            "api_key": settings.OPENAI_API_KEY,
            "model": settings.DEFAULT_AI_MODEL,
            "max_tokens": 4000,
            "temperature": 0.7
        }
    
    @staticmethod
    def get_anthropic_config() -> dict:
        return {
            "api_key": settings.ANTHROPIC_API_KEY,
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 4000
        }
    
    @staticmethod
    def get_google_config() -> dict:
        return {
            "api_key": settings.GOOGLE_AI_API_KEY,
            "model": "gemini-pro",
            "max_tokens": 4000
        }

# Logging configuration
def get_logging_config() -> dict:
    """Get logging configuration."""
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": settings.LOG_LEVEL
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": "app.log",
                "formatter": "detailed",
                "level": "DEBUG"
            }
        },
        "loggers": {
            "": {
                "handlers": ["console", "file"],
                "level": settings.LOG_LEVEL,
                "propagate": False
            },
            "uvicorn": {
                "handlers": ["console"],
                "level": "INFO",
                "propagate": False
            }
        }
    }

# Development settings override
if settings.ENVIRONMENT == "development":
    settings.DEBUG = True
    settings.DATABASE_ECHO = True
    settings.LOG_LEVEL = "DEBUG"

# Production settings override
if settings.ENVIRONMENT == "production":
    settings.DEBUG = False
    settings.DATABASE_ECHO = False
    settings.ALLOWED_HOSTS = ["terminal-plus-plus.com", "api.terminal-plus-plus.com"]
    settings.ALLOWED_ORIGINS = [
        "https://terminal-plus-plus.com",
        "https://www.terminal-plus-plus.com"
    ]