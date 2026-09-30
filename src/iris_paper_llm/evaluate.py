"""Reviewed-corpus execution and Markdown reporting."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote

from iris_paper_llm.classify import (
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    IRIS_PROMPT_SHA256,
    IRIS_PROMPT_VERSION,
    classification_key,
    pdf_sha256,
    summarize,
)
from iris_paper_llm.download import download_records
from iris_paper_llm.jsonl import read_jsonl
from iris_paper_llm.models import ResultStatus

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    import requests

DEFAULT_CASES = Path("data/eval/reviewed_cases.jsonl")
DEFAULT_OUTPUT = Path("data/eval/classification_results.jsonl")


def _expected(case: dict) -> dict:
    return {"include": case["expected_include"], "basis": case["expected_basis"]}


def case_jobs(cases_path: Path, *, case_ids: Iterable[str] | None = None, limit: int | None = None) -> Iterator[dict]:
    """`classify_jobs` input for the reviewed cases, checked against their frozen checksums."""
    cases = read_jsonl(cases_path)
    if case_ids:
        selected_ids = set(case_ids)
        cases = [case for case in cases if case["id"] in selected_ids]
    return (
        {
            "path": Path(case["path"]),
            "pdf_hash": case["pdf_sha256"],
            "verify_hash": True,
            "bibcode": case.get("bibcode"),
            "record_id": case["id"],
            "expected": _expected(case),
        }
        for case in cases[:limit]
    )


def prepare_case_pdfs(
    cases_path: Path,
    *,
    timeout: float = 60,
    session: requests.Session | None = None,
) -> dict[str, int]:
    """Acquire only missing reviewed PDFs from checksum-bearing open links."""
    grouped = defaultdict(list)
    for case in read_jsonl(cases_path):
        target = Path(case["path"])
        if target.is_file() and pdf_sha256(target) == case["pdf_sha256"]:
            continue
        bibcode = case.get("bibcode")
        if not bibcode or target.name != f"{bibcode}.pdf":
            msg = f"Missing reviewed PDF cannot be acquired automatically: {target}"
            raise ValueError(msg)
        grouped[target.parent].append(case)

    summary = {"downloaded": 0, "skipped": 0, "failed": 0, "manual_queue": 0}
    for output_dir, records in grouped.items():
        current = download_records(records, output_dir, timeout=timeout, session=session)
        for key in summary:
            summary[key] += current[key]
    return summary


def latest_results(output: Path, *, model: str, reasoning_effort: str) -> list[dict]:
    """One record per paper's latest PDF: its latest success, else its latest failure, for this configuration."""
    if not output.is_file():
        return []
    latest = {}
    for record in read_jsonl(output):
        key = classification_key(record["result"]["pdf_sha256"], model=model, reasoning_effort=reasoning_effort)
        if record["evaluation_key"] != key:
            continue
        paper = record["id"]
        has_success = (
            paper in latest
            and latest[paper]["evaluation_key"] == key
            and latest[paper]["result"]["status"] == ResultStatus.CLASSIFIED
        )
        if record["result"]["status"] == ResultStatus.CLASSIFIED or not has_success:
            latest[paper] = record
    return list(latest.values())


def _decision(include: str, basis: list[str]) -> str:
    return f"{include} ({', '.join(basis)})" if basis else include


def _ads_link(bibcode: str) -> str:
    return f"https://ui.adsabs.harvard.edu/abs/{quote(bibcode, safe='')}/abstract"


def _review_lines(records: list[dict], output: Path, library: Path) -> list[str]:
    """List YES and UNCERTAIN papers missing from the library, and write their ADS links to a text file."""
    members = set(library.read_text(encoding="utf-8").split())
    review = [
        record
        for record in records
        if (record["result"]["classification"] or {}).get("include") in {"YES", "UNCERTAIN"}
        and record["bibcode"] not in members
    ]
    lines = [
        "",
        f"## To review: YES or UNCERTAIN, not in `{library.name}` ({len(review)})",
        "",
        "| Paper | Decision | Evidence |",
        "|---|---|---|",
    ]
    links = []
    for record in review:
        classification = record["result"]["classification"]
        decision = _decision(classification["include"], classification["basis"])
        evidence = "; ".join(f"p{item['page']}: {item['reason']}" for item in classification["evidence"])
        lines.append(
            f"| [{record['bibcode']}]({_ads_link(record['bibcode'])}) | {decision} | {evidence.replace('|', '/')} |"
        )
        links.append(f"{_ads_link(record['bibcode'])}  # {decision}\n")
    output.with_name(f"{output.stem}_to_review.txt").write_text("".join(links), encoding="utf-8")
    return lines


def write_report(
    output: Path,
    *,
    cases: Path | None = None,
    library: Path | None = None,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
) -> dict[str, int]:
    """Write a compact Markdown report.

    Reviewed labels come from the current `cases` file, if given. With a `library` bibcode file, the report starts
    with the YES and UNCERTAIN papers missing from it, also written as ADS links to `<output>_to_review.txt`.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    records = latest_results(output, model=model, reasoning_effort=reasoning_effort)
    if cases is not None:
        labels = {case["id"]: _expected(case) for case in read_jsonl(cases)}
        for record in records:
            record.pop("expected", None)
            if record["id"] in labels:
                record["expected"] = labels[record["id"]]
    summary = summarize(records)
    lines = [
        "# IRIS paper classification report",
        "",
        f"- Results: {len(records)} papers",
        f"- Positive: {summary['positive']}",
        f"- Negative: {summary['negative']}",
        f"- Uncertain: {summary['uncertain']}",
        f"- Failed: {summary['failed']}",
        f"- Model: `{model}`, reasoning effort `{reasoning_effort}`",
        f"- Prompt: `{IRIS_PROMPT_VERSION}`, SHA-256 `{IRIS_PROMPT_SHA256}`",
    ]
    if library is not None:
        lines.extend(_review_lines(records, output, library))
    failed = [record for record in records if record["result"]["classification"] is None]
    if failed:
        lines.extend(["", f"## Could not be classified ({len(failed)})", "", "| Paper | Error |", "|---|---|"])
        lines.extend(f"| `{record['id']}` | {record['result']['errors'][-1].replace('|', '/')} |" for record in failed)

    reviewed = [record for record in records if "expected" in record]
    if reviewed:
        classified = [record for record in reviewed if record["result"]["classification"] is not None]
        correct = sum(
            record["result"]["classification"]["include"] == record["expected"]["include"] for record in classified
        )
        basis_matches = sum(
            set(record["result"]["classification"]["basis"]) == set(record["expected"]["basis"])
            for record in classified
        )
        lines.extend(
            [
                "",
                "| Include correct | Include incorrect | Failed | Basis exact match |",
                "|---:|---:|---:|---:|",
                (
                    f"| {correct} | {len(classified) - correct} | {len(reviewed) - len(classified)} "
                    f"| {basis_matches} of {len(classified)} |"
                ),
            ]
        )

    lines.extend(["", "| Paper | Expected | Predicted | Basis | Status |", "|---|---|---|---|---|"])
    for record in records:
        expected = record.get("expected")
        classification = record["result"]["classification"] or {}
        lines.append(
            f"| `{record['id']}` | {_decision(expected['include'], expected['basis']) if expected else '-'} "
            f"| {classification.get('include', '-')} | {', '.join(classification.get('basis', [])) or '-'} "
            f"| {record['result']['status']} |"
        )
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
