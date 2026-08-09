"""Opt-in library logging without modifying application-wide configuration."""

import logging

LOGGER_NAME = "voice_to_text"
_logger = logging.getLogger(LOGGER_NAME)
_logger.addHandler(logging.NullHandler())


def get_logger(component: str | None = None) -> logging.Logger:
    return logging.getLogger(LOGGER_NAME if not component else f"{LOGGER_NAME}.{component}")


def configure_logging(level: int = logging.INFO, handler: logging.Handler | None = None) -> None:
    """Configure only this library; parent applications may instead configure it themselves."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    if handler is not None and handler not in logger.handlers:
        logger.addHandler(handler)
