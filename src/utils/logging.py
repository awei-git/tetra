import logging
import sys
from loguru import logger
from typing import Any
import json
from datetime import datetime


class InterceptHandler(logging.Handler):
    """Intercept standard logging messages and route them to loguru"""
    
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def serialize_record(record: dict[str, Any]) -> str:
    """Serialize log record to JSON"""
    subset = {
        "timestamp": record["time"].timestamp(),
        "message": record["message"],
        "level": record["level"].name,
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
    }
    
    # Add exception info if present
    if record["exception"] is not None:
        subset["exception"] = {
            "type": record["exception"].type.__name__,
            "value": str(record["exception"].value),
            "traceback": record["exception"].traceback.format(),
        }
    
    # Add extra fields
    subset.update(record.get("extra", {}))
    
    return json.dumps(subset)


def setup_logging(log_level: str = None) -> Any:
    """Configure logging for the application"""
    from config import settings
    
    log_level = log_level or settings.log_level
    
    # Remove default logger
    logger.remove()
    
    # Add console logger with custom format
    if settings.app_env == "development":
        # Human-readable format for development
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=log_level,
            colorize=True,
        )
    else:
        # JSON format for production
        logger.add(
            sys.stdout,
            format=serialize_record,
            level=log_level,
            serialize=True,
        )
    
    # Add file logger
    logger.add(
        "logs/tetra_{time}.log",
        rotation="1 day",
        retention="7 days",
        level=log_level,
        format=serialize_record if settings.app_env != "development" else "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        serialize=True if settings.app_env != "development" else False,
    )
    
    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0)
    
    # Intercept uvicorn logs
    for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
        logging.getLogger(logger_name).handlers = [InterceptHandler()]
    
    return logger