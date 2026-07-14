"""ADS search and metadata acquisition for the CLI pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from paper_data_linking.data.bibcode_service import BibcodeService
from paper_data_linking.data.metadata_service import MetadataService

IRIS_INSTRUMENT_BIBCODE = "2014SoPh..289.2733D"


def iris_query(year: int) -> str:
    """Build the established refereed IRIS discovery query for one year."""
    return (
        '(=full:"Interface Region Imaging Spectrograph" '
        f"OR citations(bibcode:{IRIS_INSTRUMENT_BIBCODE})) "
        f'+ property:refereed + doctype:"Article" + pubdate:[{year}-01 TO {year}-12]'
    )


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary_path = Path(stream.name)
            for record in records:
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def search_papers(
    query: str,
    output: Path,
    *,
    api_token: str | None = None,
    limit: int = 2000,
    force: bool = False,
) -> dict[str, int]:
    """Search ADS, fetch download metadata, and write one durable JSONL file."""
    if limit <= 0:
        msg = "limit must be positive"
        raise ValueError(msg)
    if output.is_file() and not force:
        records = sum(1 for line in output.read_text(encoding="utf-8").splitlines() if line.strip())
        return {"found": records, "written": 0, "skipped": records}

    api_token = api_token or os.getenv("ADS_TOKEN")
    if not api_token:
        msg = "ADS_TOKEN or --api-token is required"
        raise ValueError(msg)
    bibcodes = BibcodeService(api_token).get_bibcodes_by_query(query, rows=limit)
    metadata = MetadataService(api_token).get_metadata_for_bibcodes(bibcodes)
    records = [record.to_dict() for record in metadata]
    _write_jsonl(output, records)
    return {"found": len(bibcodes), "written": len(records), "skipped": 0}
