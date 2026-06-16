"""Centralized logger setup for APEX Optimizer."""
import logging
import logging.handlers
import colorlog
from config.settings import LOGGING


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOGGING["level"], logging.INFO))
    logger.propagate = False   # prevent double-emit via parent loggers

    # Console handler with color
    ch = colorlog.StreamHandler()
    ch.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s [%(name)s] %(levelname)s%(reset)s  %(message)s",
        datefmt="%H:%M:%S",
        log_colors={
            "DEBUG":    "cyan",
            "INFO":     "green",
            "WARNING":  "yellow",
            "ERROR":    "red",
            "CRITICAL": "bold_red",
        },
    ))
    logger.addHandler(ch)

    # Rotating file handler
    fh = logging.handlers.RotatingFileHandler(
        LOGGING["file"],
        maxBytes=LOGGING["max_bytes"],
        backupCount=LOGGING["backup_count"],
        encoding="utf-8",
    )
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)

    return logger
