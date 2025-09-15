import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

import tiktoken
from pathlib import Path

from langchain_core.documents import Document
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
    acronyms, full_names = zip(*acronym_expansion_list)
    # Check for an exact match with an acronym
    if inst_str in acronyms:
        return inst_str
    # If no exact match was found, perform fuzzy matching
    else:
        # Get best match with acronyms and its score
        best_match_acronym, score_acronym = process.extractOne(
            inst_str, acronyms, scorer=fuzz.token_sort_ratio
        )
        # Get best match with full names and its score
        best_match_full_name, score_full_name = process.extractOne(
            inst_str, full_names, scorer=fuzz.token_sort_ratio
        )
        # If both scores are below the threshold, return the original string
        if score_acronym < threshold and score_full_name < threshold:
            return inst_str
        # Else, return the acronym corresponding to the best match
        else:
            if score_acronym > score_full_name:
                return best_match_acronym
            else:
                index = full_names.index(best_match_full_name)
                return acronyms[index]


def get_correct_instruments(
    instruments, metadata,
):
    if metadata is None:
        return instruments
    acronym_expansion_list = [(k, v['detail']) for k, v in metadata.items()]
    if acronym_expansion_list is None:
        acronym_expansion_list = INSTRUMENT_ACRONYM_EXPANSIONS
    corrected_instruments = [
        get_acronym(inst, acronym_expansion_list) for inst in instruments
    ]
    return corrected_instruments


def to_jsonlines(docs, outfile):
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("w+") as f0:
        for d in docs:
            json_str = json.dumps(d)
            f0.write(json_str)
            f0.write("\n")
    return outfile


def write_pdf(content, dir_loc, bibcode):
    dir_loc = Path(dir_loc)
    dir_loc.mkdir(parents=True, exist_ok=True)
    if content is not None:
        with (dir_loc / f"{bibcode}.pdf").open("wb") as f0:
            f0.write(content)
        return 1
    else:
        return 0

def read_local_pdf(file_path):
    with open(file_path, "rb") as f:
        content = f.read()
    return content

def validate_pdfs(dir_loc):
    dir_loc = Path(dir_loc)
    pdf_files = list(dir_loc.glob("*.pdf"))
    valid_count = 0
    invalid_files = []
    for pdf_file in pdf_files:
        try:
            with open(pdf_file, "rb") as f:
                header = f.read(5)
                if header == b"%PDF-":
                    valid_count += 1
                else:
                    invalid_files.append(pdf_file.name)
        except Exception as e:
            invalid_files.append(pdf_file.name)
    total_files = len(pdf_files)
    if total_files > 0:
        print(f"Validated {valid_count}/{total_files} PDF files.")
    if invalid_files:
        print(f"Invalid PDF files: {invalid_files}")
        print("Removing invalid PDF files...")
        # Remove invalid files and add it to failed_bibcodes.txt
        failed_bibcodes_file = dir_loc / "failed_bibcodes.txt"
        for invalid_file in invalid_files:
            with failed_bibcodes_file.open('a') as f:
                f.write(f"{invalid_file.split('.pdf')[0]}\n")
            (dir_loc / invalid_file).unlink()


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
    with open(infile, "r") as f0:
        bibcodes = f0.readlines()
    bibcodes = [b.strip() for b in bibcodes if b.strip() != ""]
    return bibcodes


def get_stage_message(stage=0):
    in_progress = "⏳"
    complete = "✅"
    stages = ["parsing", "embedding", "analyzing"]
    messages = []
    for i, s in enumerate(stages):
        if i <= stage:
            marker = complete
        else:
            marker = in_progress
        messages.append(f"{s}{marker}")
    message = " › ".join(messages)
    return message


class PluginRecord(BaseModel):
    analyzer: str
    passed_heuristic_filter: bool
    data: Optional[Dict[str, Any]] = None
    analysis: Optional[str] = None
    relevant_indices: Optional[List[int]] = None

class ContentAnalyzerRecord(BaseModel):
    docs: List[Document]
    ocr_status: bool
    records: List[PluginRecord]



def get_time():
    timestamp = datetime.now()
    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    return timestamp_str
