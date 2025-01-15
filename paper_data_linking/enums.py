from enum import Enum
from pathlib import Path
from paper_data_linking.settings import DATA_DIR_NAME

DATA_DIR = Path(__file__).absolute().parent.parent / "data"
if DATA_DIR_NAME is not None:
    EXP_DIR = DATA_DIR / DATA_DIR_NAME
else:
    EXP_DIR = DATA_DIR / "soho"
MEDIA_DIR = Path(__file__).absolute().parent.parent / "media"
MODEL_DIR = Path(__file__).absolute().parent.parent / "models"


class Locations(Enum):
    RESUME_TOKEN = DATA_DIR / "soho" / "resume_token.txt"
    BIBCODES = EXP_DIR / "bibcodes"
    SOHO_BIBCODES_TXT = DATA_DIR / "soho" / "bibcodes.txt"
    SOHO_BIBCODES_DIR = DATA_DIR / "soho" / "bibcodes"
    NON_SOHO_BIBCODES_TXT = DATA_DIR / "non_soho" / "bibcodes.txt"
    ALL_BIBCODES_TXT = EXP_DIR / "bibcodes.txt"
    PAPERS = EXP_DIR / "papers"
    TEXTS = EXP_DIR / "text"
    PDF = EXP_DIR / "pdf"
    LATEX = EXP_DIR / "latex"
    METADATA_RECORDS = EXP_DIR / "metadata_records.jsonl"
    BIBCODE_URLS = EXP_DIR / "bibcode_urls.jsonl"
    COUNT_DF = EXP_DIR / "counts.csv"
    COUNT_PLOT = MEDIA_DIR / "count_plot.png"
    PUB_PLOT = MEDIA_DIR / "pub_plot.png"
    SEGMENT_DF = EXP_DIR / "segments.csv"
    FAILED_URLS = EXP_DIR / "failed_urls.jsonl"
    CHROMEDRIVER_LOG = EXP_DIR / "chromedriver_log.txt"
    VECTORSTORES_DIR = EXP_DIR / "vectorstores"


class ModelPaths(Enum):
    ALPACA_7B = MODEL_DIR / "ggml-alpaca-7b-q4.bin"


class DataFields(Enum):
    BIBCODE = "bibcode"
    TEXT = "text"
    TOKEN_COUNT = "token_count"
    VALID = "valid"

