import logging
import os
from pathlib import Path

from dotenv import load_dotenv, find_dotenv

log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)

logging.basicConfig(
    level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

env_file = find_dotenv()
load_dotenv()

THIS_DIR = Path(__file__).resolve().parent
STATIC_DIR = THIS_DIR / "static"
TEMPLATES_DIR = THIS_DIR / "templates"
DB_DIR = THIS_DIR / "db"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REDIS_HOST = os.getenv("REDIS_HOST")
CELERY_ALWAYS_EAGER = bool(int(os.getenv("CELERY_ALWAYS_EAGER", 0)))
CONFIG_DIR = THIS_DIR.parent / "config"
