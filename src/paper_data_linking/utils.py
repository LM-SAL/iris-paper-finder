import os
import re
import json
from typing import Any
from pathlib import Path
from datetime import UTC, datetime

import aiofiles
from langchain_core.documents import Document
from pydantic import BaseModel
from pypdf import PdfReader
from thefuzz import fuzz, process

INSTRUMENT_DOI_DICT = {
    "CDS": "https://doi.org/10.5270/esa-50ehv09",
    "CELIAS": "https://doi.org/10.5270/esa-iey8z2h",
    "COSTEP": "https://doi.org/10.5270/esa-yrl3swm",
    "EIT": "https://doi.org/10.5270/esa-yb6hwf9",
    "ERNE": "https://doi.org/10.5270/esa-xch97ty",
    "GOLF": "https://doi.org/10.5270/esa-ls55aku",
    "LASCO": "https://doi.org/10.5270/esa-4qhsiaj",
    "MDI": "https://doi.org/10.5270/esa-9kpubs2",
    "SUMER": "https://doi.org/10.5270/esa-shl1wiw",
    "SWAN": "https://doi.org/10.5270/esa-k1pfbpn",
    "UVCS": "https://doi.org/10.5270/esa-b4hhucr",
    "VIRGO": "https://doi.org/10.5270/esa-vief1sn",
}


INSTRUMENT_ACRONYM_EXPANSIONS = [
    ("CDS", "Coronal Diagnostic Spectrometer"),
    ("CELIAS", "Charge, Element, and Isotope Analysis System"),
    ("COSTEP", "Comprehensive Suprathermal and Energetic Particle Analyzer"),
    ("EIT", "Extreme ultraviolet Imaging Telescope"),
    ("ERNE", "Energetic and Relativistic Nuclei and Electron"),
    ("GOLF", "Global Oscillations at Low Frequencies"),
    ("LASCO", "Large Angle and Spectrometric Coronagraph"),
    ("MDI", "Michelson Doppler Imager"),
    ("SUMER", "Solar Ultraviolet Measurements of Emitted Radiation"),
    ("SWAN", "Solar Wind Anisotropies"),
    ("UVCS", "Ultraviolet Coronagraph Spectrometer"),
    ("VIRGO", "Variability of solar IRradiance and Gravity Oscillations"),
]


def get_acronym(inst_str, acronym_expansion_list, threshold=70):
    # Create separate lists of acronyms and full names for fuzzy matching
    acronyms, full_names = zip(*acronym_expansion_list, strict=False)
    # Check for an exact match with an acronym
    if inst_str in acronyms:
        return inst_str
    # If no exact match was found, perform fuzzy matching
    # Get best match with acronyms and its score
    best_match_acronym, score_acronym = process.extractOne(inst_str, acronyms, scorer=fuzz.token_sort_ratio)
    # Get best match with full names and its score
    best_match_full_name, score_full_name = process.extractOne(inst_str, full_names, scorer=fuzz.token_sort_ratio)
    # If both scores are below the threshold, return the original string
    if score_acronym < threshold and score_full_name < threshold:
        return inst_str
    # Else, return the acronym corresponding to the best match
    if score_acronym > score_full_name:
        return best_match_acronym
    index = full_names.index(best_match_full_name)
    return acronyms[index]


def get_correct_instruments(
    instruments,
    metadata,
):
    if metadata is None:
        return instruments
    acronym_expansion_list = [(k, v["detail"]) for k, v in metadata.items()]
    if acronym_expansion_list is None:
        acronym_expansion_list = INSTRUMENT_ACRONYM_EXPANSIONS
    return [get_acronym(inst, acronym_expansion_list) for inst in instruments]


def to_jsonlines(docs, outfile):
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("w+") as f0:
        for d in docs:
            json_str = json.dumps(d)
            f0.write(json_str)
            f0.write("\n")
    return outfile


async def write_pdf(content: bytes, dir_loc: Path, bibcode: str) -> None:
    dir_loc = Path(dir_loc)
    dir_loc.mkdir(parents=True, exist_ok=True)
    path = dir_loc / f"{bibcode}.pdf"
    async with aiofiles.open(path, "wb") as f:
        await f.write(content)
        return 1
    return 0


async def append_failed_bibcode(path: Path, bibcode: str) -> None:
    async with aiofiles.open(path, "a") as f:
        await f.write(f"{bibcode}\n")


def read_local_pdf(file_path):
    with open(file_path, "rb") as f:
        return f.read()


def validate_pdfs(dir_loc, min_size: int = 8 * 1024) -> None:
    dir_loc = Path(dir_loc)
    pdf_files = list(dir_loc.glob("*.pdf"))
    valid_count = 0
    invalid_files: list[str] = []

    def _is_valid_pdf(path: Path) -> bool:  # NOQA:PLR0911
        size = path.stat().st_size
        if size < min_size:
            return False
        with path.open("rb") as f:
            header = f.read(8)
            if not header.startswith(b"%PDF-"):
                return False
            tail_size = min(2048, size)
            try:
                f.seek(-tail_size, os.SEEK_END)
            except OSError:
                f.seek(0)
            tail = f.read()
            if b"%%EOF" not in tail:
                return False
        try:
            with path.open("rb") as f:
                reader = PdfReader(f, strict=False)
                # Require at least 2 pages
                if len(reader.pages) < 2:
                    return False
                text = ""
                for page in reader.pages[:2]:
                    page_text = page.extract_text() or ""
                    text += page_text.lower()

                if "abstract" not in text and "references" not in text:
                    return False
        except Exception:
            return False
        return True

    for pdf_file in pdf_files:
        try:
            if _is_valid_pdf(pdf_file):
                valid_count += 1
            else:
                invalid_files.append(pdf_file.name)
        except Exception:
            invalid_files.append(pdf_file.name)
    if invalid_files:
        failed_bibcodes_file = dir_loc / "failed_bibcodes.txt"
        for invalid_file in invalid_files:
            bibcode = invalid_file.split(".pdf")[0]
            with failed_bibcodes_file.open("a") as f:
                f.write(f"{bibcode}\n")
            (dir_loc / invalid_file).unlink(missing_ok=True)


def clean_collection_name_for_chroma(input_string):
    # Convert the input string to lowercase
    input_string = input_string.lower()
    # Remove invalid characters
    cleaned_string = re.sub(r"[^a-z0-9._-]", "", input_string)
    # Remove consecutive periods
    cleaned_string = re.sub(r"\.{2,}", ".", cleaned_string)
    # Trim string length to fit the 3-63 characters limit
    cleaned_string = cleaned_string[:63]
    # Ensure the string starts and ends with an alphanumeric character
    cleaned_string = re.sub(r"^[^a-z0-9]+", "", cleaned_string)
    cleaned_string = re.sub(r"[^a-z0-9]+$", "", cleaned_string)
    # Replace periods with underscores
    cleaned_string = re.sub(r"\.", "_", cleaned_string)
    # If the string is still shorter than 3 characters, pad with 'a's
    while len(cleaned_string) < 3:
        cleaned_string += "a"
    return cleaned_string


def load_bibcodes_list(infile):
    with open(infile) as f0:
        bibcodes = f0.readlines()
    return [b.strip() for b in bibcodes if b.strip() != ""]


def get_stage_message(stage=0):
    in_progress = "⏳"
    complete = "✅"
    stages = ["parsing", "embedding", "analyzing"]
    messages = []
    for i, s in enumerate(stages):
        marker = complete if i <= stage else in_progress
        messages.append(f"{s}{marker}")
    return " > ".join(messages)


class PluginRecord(BaseModel):
    analyzer: str
    passed_heuristic_filter: bool
    data: dict[str, Any] | None = None
    analysis: str | None = None
    relevant_indices: list[int] | None = None


class ContentAnalyzerRecord(BaseModel):
    docs: list[Document]
    ocr_status: bool
    records: list[PluginRecord]


def get_time():
    timestamp = datetime.now(tz=UTC)
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")
