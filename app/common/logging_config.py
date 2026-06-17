import logging
import logging.config
import os
from typing import Any, Dict

_logging_configured = False


def resolve_log_dir() -> str:
    env_dir = os.getenv("OCR_LOG_DIR")
    if env_dir and env_dir.strip():
        log_dir = env_dir.strip()
    else:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def build_uvicorn_log_config(log_dir: str) -> Dict[str, Any]:
    service_log = os.path.join(log_dir, "ocr-service.log")
    access_log = os.path.join(log_dir, "access.log")

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "access": {
                "format": '%(asctime)s [ACCESS] %(client_addr)s - "%(request_line)s" %(status_code)s',
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "service_file": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "formatter": "default",
                "filename": service_log,
                "when": "midnight",
                "backupCount": 14,
                "encoding": "utf-8",
            },
            "access_file": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "formatter": "access",
                "filename": access_log,
                "when": "midnight",
                "backupCount": 14,
                "encoding": "utf-8",
            },
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
            },
        },
        "loggers": {
            "uvicorn": {
                "handlers": ["service_file", "console"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.error": {
                "handlers": ["service_file", "console"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["access_file", "console"],
                "level": "INFO",
                "propagate": False,
            },
            "app": {
                "handlers": ["service_file", "console"],
                "level": "INFO",
                "propagate": False,
            },
            "ppocr": {
                "handlers": ["service_file", "console"],
                "level": "WARNING",
                "propagate": False,
            },
        },
        "root": {
            "handlers": ["service_file", "console"],
            "level": "INFO",
        },
    }


def ensure_logging_configured() -> None:
    """Apply file/console handlers once (works with or without run_server.py)."""
    global _logging_configured
    if _logging_configured:
        return

    app_logger = logging.getLogger("app")
    if app_logger.handlers:
        _logging_configured = True
        return

    log_dir = resolve_log_dir()
    logging.config.dictConfig(build_uvicorn_log_config(log_dir))
    _logging_configured = True


def get_logger(name: str = "app") -> logging.Logger:
    ensure_logging_configured()
    return logging.getLogger(name)
