"""Reviewed-corpus execution and Markdown reporting."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from paper_data_linking.classify import (
    DEFAULT_MODEL,
    IRIS_PROMPT_SHA256,
    append_record,
    classification_key,
    classify_pdf,
    create_openai_client,
    load_successful_keys,
    pdf_sha256,
)
from paper_data_linking.download import download_records
from paper_data_linking.models import Decision, PaperResult, RetrievalMode
from paper_data_linking.retrieval import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DEFAULT_TOP_K

if TYPE_CHECKING:
    from collections.abc import Iterable

DEFAULT_CASES = Path("data/eval/reviewed_cases.jsonl")
DEFAULT_OUTPUT = Path("data/eval/classification_results.jsonl")


def load_cases(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def case_path(case: dict, *, base_dir: Path) -> Path:
    path = Path(case["path"])
    return path if path.is_absolute() else base_dir / path


def expected_classification(case: dict) -> dict[str, str]:
    return {
        "overall": case["expected_overall"],
        "observational_use": case["expected_observational_use"],
        "synthetic_use": case["expected_synthetic_use"],
        "synthetic_connection": case["synthetic_iris_connection"],
        "review_only": case["expected_review_only"],
    }


def _add_result(summary: dict[str, int], record: dict) -> None:
    result = PaperResult.model_validate(record["result"])
    if result.classification is None:
        summary["failed"] += 1
        return
    key = {
        Decision.YES: "positive",
        Decision.NO: "negative",
        Decision.UNCERTAIN: "uncertain",
    }[result.classification.overall]
    summary[key] += 1


def classify_cases(
    cases_path: Path,
    output: Path,
    *,
    base_dir: Path | None = None,
    case_ids: Iterable[str] | None = None,
    limit: int | None = None,
    force: bool = False,
    model: str = DEFAULT_MODEL,
    retrieval_mode: RetrievalMode = RetrievalMode.AUTO,
    top_k: int = DEFAULT_TOP_K,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ocr_fallback: bool = True,
    client: object | None = None,
    embedder: object | None = None,
) -> dict[str, int]:
    """Run reviewed cases through the same one-PDF function as the CLI."""
    base_dir = base_dir or Path.cwd()
    cases = load_cases(cases_path)
    if case_ids:
        selected_ids = set(case_ids)
        cases = [case for case in cases if case["id"] in selected_ids]
    if limit is not None:
        cases = cases[:limit]

    successful = set() if force else load_successful_keys(output)
    summary = {"positive": 0, "negative": 0, "uncertain": 0, "failed": 0, "skipped": 0}
    shared_client = client
    for case in cases:
        key = classification_key(
            case["pdf_sha256"],
            model=model,
            retrieval_mode=retrieval_mode,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
        )
        if key in successful:
            summary["skipped"] += 1
            continue
        path = case_path(case, base_dir=base_dir)
        if path.is_file() and shared_client is None:
            shared_client = create_openai_client()
        classified = classify_pdf(
            path,
            bibcode=case.get("bibcode"),
            record_id=case["id"],
            expected_sha256=case["pdf_sha256"],
            model=model,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            ocr_fallback=ocr_fallback,
            client=shared_client,
            embedder=embedder,
        )
        record = {
            "evaluation_key": classified["evaluation_key"],
            "id": classified["id"],
            "bibcode": classified["bibcode"],
            "expected": expected_classification(case),
            "retrieval": classified["retrieval"],
            "result": classified["result"],
        }
        append_record(output, record)
        _add_result(summary, record)
    return summary


def prepare_case_pdfs(
    cases_path: Path,
    *,
    base_dir: Path | None = None,
    timeout: float = 60,
    browser_fallback: bool = False,
    session: object | None = None,
) -> dict[str, int]:
    """Acquire only missing reviewed PDFs from checksum-bearing open links."""
    base_dir = base_dir or Path.cwd()
    grouped = defaultdict(list)
    for case in load_cases(cases_path):
        target = case_path(case, base_dir=base_dir)
        if target.is_file() and pdf_sha256(target) == case["pdf_sha256"]:
            continue
        bibcode = case.get("bibcode")
        if not bibcode or target.name != f"{bibcode}.pdf":
            msg = f"Missing reviewed PDF cannot be acquired automatically: {target}"
            raise ValueError(msg)
        grouped[target.parent].append(case)

    summary = {"downloaded": 0, "skipped": 0, "failed": 0, "manual_queue": 0}
    for output_dir, records in grouped.items():
        current = download_records(
            records,
            output_dir,
            timeout=timeout,
            browser_fallback=browser_fallback,
            session=session,
        )
        for key in summary:
            summary[key] += current[key]
    return summary


def _matching_results(
    output: Path,
    *,
    model: str | None = None,
    retrieval_mode: RetrievalMode | None = None,
    top_k: int | None = None,
) -> list[dict]:
    if not output.is_file():
        return []
    latest = {}
    for line in output.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        retrieval = record["retrieval"]
        if model is not None:
            expected_key = classification_key(
                record["result"]["pdf_sha256"],
                model=model,
                retrieval_mode=retrieval_mode or retrieval["mode"],
                chunk_size=retrieval["chunk_size"],
                chunk_overlap=retrieval["chunk_overlap"],
                top_k=top_k or retrieval["top_k"],
            )
            if record["evaluation_key"] != expected_key:
                continue
        elif (retrieval_mode is not None and retrieval["mode"] != retrieval_mode) or (
            top_k is not None and retrieval["top_k"] != top_k
        ):
            continue
        latest[record["evaluation_key"]] = record
    return list(latest.values())


def prediction(record: dict, field: str) -> str | None:
    result = PaperResult.model_validate(record["result"])
    if result.classification is None:
        return None
    if field == "overall":
        return result.classification.overall
    return getattr(result.classification, field)


def summarize_results(records: Iterable[dict]) -> dict[str, int]:
    summary = {"positive": 0, "negative": 0, "uncertain": 0, "failed": 0, "skipped": 0}
    for record in records:
        _add_result(summary, record)
    return summary


def write_report(
    output: Path,
    *,
    model: str | None = None,
    retrieval_mode: RetrievalMode | None = None,
    top_k: int | None = None,
) -> dict[str, int]:
    """Write a compact report for reviewed or ordinary classification JSONL."""
    records = _matching_results(output, model=model, retrieval_mode=retrieval_mode, top_k=top_k)
    summary = summarize_results(records)
    lines = [
        "# IRIS paper classification report",
        "",
        f"- Results: {len(records)} papers",
        f"- Positive: {summary['positive']}",
        f"- Negative: {summary['negative']}",
        f"- Uncertain: {summary['uncertain']}",
        f"- Failed: {summary['failed']}",
    ]
    if model is not None:
        lines.extend([f"- Model: `{model}`", f"- Prompt SHA-256: `{IRIS_PROMPT_SHA256}`"])

    reviewed = [record for record in records if "expected" in record]
    if reviewed:
        lines.extend(
            [
                "",
                "| Field | Correct | Incorrect | Failed |",
                "|---|---:|---:|---:|",
            ]
        )
        fields = ("overall", "observational_use", "synthetic_use", "synthetic_connection", "review_only")
        for field in fields:
            correct = incorrect = failed = 0
            for record in reviewed:
                actual = prediction(record, field)
                if actual is None:
                    failed += 1
                elif actual == record["expected"][field]:
                    correct += 1
                else:
                    incorrect += 1
            lines.append(f"| `{field}` | {correct} | {incorrect} | {failed} |")

    lines.extend(["", "| Paper | Predicted | Status |", "|---|---|---|"])
    for record in records:
        predicted = prediction(record, "overall") or "-"
        lines.append(f"| `{record['id']}` | {predicted} | {record['result']['status']} |")
    output.with_suffix(".md").parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
