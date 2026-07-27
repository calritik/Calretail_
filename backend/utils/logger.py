"""
CalRetail — Logger utility (loguru-based)
"""
import sys
from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
    level="INFO",
)
logger.add(
    "logs/calretail.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
)

__all__ = ["logger"]
