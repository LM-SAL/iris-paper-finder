"""ADS search and download metadata for the CLI pipeline."""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

import requests

from iris_paper_llm.jsonl import write_jsonl

if TYPE_CHECKING:
    from pathlib import Path

# The single place to change when moving from ADS to SciX (scixplorer).
ADS_API = "https://api.adsabs.harvard.edu/v1"
ADS_MAX_ROWS = 2000
IRIS_INSTRUMENT_BIBCODE = "2014SoPh..289.2733D"
logger = logging.getLogger(__name__)


def iris_query(year: int) -> str:
    """Build the refereed IRIS discovery query for one publication year."""
    return (
        "("
        # Full instrument name, exact phrase (no synonym expansion).
        '=full:"Interface Region Imaging Spectrograph"'
        # Papers citing the IRIS instrument paper.
        f" OR citations(bibcode:{IRIS_INSTRUMENT_BIBCODE})"
        # Acronym only: unprefixed uppercase "IRIS" is matched case-sensitively; the
        # IRIS-specific channels/lines and the astronomy database drop TMT/IRAS/eye-iris noise.
        ' OR (full:"IRIS" AND full:("slit-jaw" OR "slit jaw" OR "SJI" OR "Mg II" OR "Si IV" OR "C II")'
        " AND database:astronomy)"
        # Standard IRIS data acknowledgement sentence.
        ' OR full:("IRIS is a NASA small explorer" OR "developed and operated by LMSAL")'
        # Synthetic/modelled observables in IRIS passbands, solar or stellar, IRIS need not be named.
        ' OR (abs:("Mg II" OR "Si IV" OR "C II" OR "O IV" OR "Fe XII 1349" OR "Fe XXI" OR "Cl I"'
        ' OR "NUV continuum" OR "near-ultraviolet continuum" OR "near-UV continuum")'
        ' AND abs:(synthetic OR synthesized OR synthesised OR synthesis OR "forward model"'
        ' OR "forward modeling" OR "forward modelling" OR "radiative transfer" OR "non-LTE" OR NLTE'
        ' OR "radiative hydrodynamic" OR "radiative-hydrodynamic" OR "radiation hydrodynamic")'
        ' AND abs:(chromosphere OR chromospheric OR "transition region" OR flare OR flares)'
        " AND database:astronomy)"
        ")"
        f' + property:refereed + doctype:"Article" + pubdate:[{year}-01 TO {year}-12]'
    )


def _download_record(document: dict) -> dict:
    record = dict(document)
    record["links_data"] = [
        json.loads(link) if isinstance(link, str) else link for link in record.get("links_data", []) or []
    ]
    return record


def search_papers(query: str, output: Path, *, limit: int = 2000) -> dict[str, int]:
    """Query ADS page by page and replace the metadata JSONL with bibcodes, link data, DOIs and abstracts."""
    if limit <= 0:
        msg = "limit must be positive"
        raise ValueError(msg)
    api_token = os.getenv("ADS_TOKEN")
    if not api_token:
        msg = "ADS_TOKEN is not set; add it to .env"
        raise ValueError(msg)

    records: list[dict] = []
    while True:
        response = requests.get(
            f"{ADS_API}/search/query",
            params={
                "q": query,
                "fl": "bibcode,links_data,doi,abstract",
                "sort": "bibcode asc",
                "start": len(records),
                "rows": min(ADS_MAX_ROWS, limit - len(records)),
            },
            headers={"Authorization": f"Bearer {api_token}"},
            timeout=60,
        )
        response.raise_for_status()
        page = response.json()["response"]
        found = page["numFound"]
        records.extend(_download_record(document) for document in page["docs"])
        if len(records) >= min(found, limit):
            break
        if not page["docs"]:
            msg = f"ADS returned no records at offset {len(records)} of {found}"
            raise RuntimeError(msg)
    if found > limit:
        logger.warning("ADS matched %d papers; --limit %d kept the first %d by bibcode", found, limit, len(records))
    write_jsonl(output, records)
    return {"found": found, "written": len(records)}
