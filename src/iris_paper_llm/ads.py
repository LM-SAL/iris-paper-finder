"""ADS search and metadata acquisition for the CLI pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import requests

IRIS_INSTRUMENT_BIBCODE = "2014SoPh..289.2733D"


def _bibcodes_by_query(query: str, token: str, limit: int) -> list[str]:
    response = requests.get(
        "https://api.adsabs.harvard.edu/v1/search/query",
        params={"q": query, "fl": "bibcode", "rows": limit},
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    response.raise_for_status()
    return [document["bibcode"] for document in response.json()["response"]["docs"]]


def _metadata_for_bibcodes(bibcodes: list[str], token: str) -> list[dict]:
    if not bibcodes:
        return []
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "big-query/csv",
    }
    payload = "bibcode\n" + "\n".join(bibcodes)
    records = []
    start = 0
    total = len(bibcodes)
    while start < total:
        response = requests.post(
            "https://api.adsabs.harvard.edu/v1/search/bigquery",
            params={"q": "*:*", "fl": "bibcode,links_data", "rows": 2000, "start": start},
            headers=headers,
            data=payload,
            timeout=30,
        )
        response.raise_for_status()
        page = response.json()["response"]
        total = page["numFound"]
        documents = page["docs"]
        if not documents and start < total:
            msg = f"ADS returned no metadata records at offset {start} of {total}"
            raise RuntimeError(msg)
        for document in documents:
            record = dict(document)
            record["links_data"] = [
                json.loads(link) if isinstance(link, str) else link for link in record.get("links_data", []) or []
            ]
            record.setdefault("pdf_links", [])
            records.append(record)
        start = page["start"] + len(documents)
    return records


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
    bibcodes = _bibcodes_by_query(query, api_token, limit)
    records = _metadata_for_bibcodes(bibcodes, api_token)
    _write_jsonl(output, records)
    return {"found": len(bibcodes), "written": len(records), "skipped": 0}
