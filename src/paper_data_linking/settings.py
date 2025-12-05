import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMEDRIVER_PATH = os.getenv("CHROMEDRIVER_PATH")
DATA_DIR_NAME = os.getenv("DATA_DIR_NAME")
ADS_TOKEN = os.getenv("ADS_TOKEN")
