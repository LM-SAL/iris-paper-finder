"""Run checkpointed structured classification on reviewed PDFs."""

# ruff: noqa: T201

from __future__ import annotations

import json
import hashlib
import argparse
from pathlib import Path

from dotenv import load_dotenv

from paper_data_linking.classify import IRIS_PROMPT_SHA256, PIPELINE_VERSION, classify_paper
from paper_data_linking.models import PaperResult, ResultStatus, RetrievalMode
from paper_data_linking.retrieval import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_TOP_K,
    chunk_pdf,
    retrieve_chunks,
)

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "data/eval/reviewed_cases.jsonl"
DEFAULT_OUTPUT = ROOT / "data/eval/classification_results.jsonl"
DEFAULT_MODEL = "gpt-5-mini-2025-08-07"


def load_cases() -> list[dict]:
    return [json.loads(line) for line in CASES_PATH.read_text().splitlines()]


def evaluation_key(case: dict, *, model: str, mode: RetrievalMode, top_k: int) -> str:
    values = {
        "pdf_sha256": case["pdf_sha256"],
        "prompt_sha256": IRIS_PROMPT_SHA256,
        "pipeline_version": PIPELINE_VERSION,
        "model": model,
        "retrieval_mode": mode,
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
        "top_k": top_k,
    }
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def load_successful_keys(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    successful = set()
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record["result"]["status"] == ResultStatus.CLASSIFIED:
            successful.add(record["evaluation_key"])
    return successful


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(record, separators=(",", ":")) + "\n")


def classify_case(case: dict, *, model: str, mode: RetrievalMode, top_k: int) -> dict:
    chunks, used_ocr = chunk_pdf(ROOT / case["path"])
    retrieval = retrieve_chunks(chunks, mode=mode, top_k=top_k)
    result = classify_paper(
        retrieval.selected,
        bibcode=case["bibcode"],
        pdf_sha256=case["pdf_sha256"],
        retrieval_mode=mode,
        model=model,
    )
    return {
        "evaluation_key": evaluation_key(case, model=model, mode=mode, top_k=top_k),
        "id": case["id"],
        "bibcode": case["bibcode"],
        "expected": {
            "overall": case["expected_overall"],
            "observational_use": case["expected_observational_use"],
            "synthetic_use": case["expected_synthetic_use"],
            "synthetic_connection": case["synthetic_iris_connection"],
            "review_only": case["expected_review_only"],
        },
        "retrieval": {
            "mode": mode,
            "top_k": top_k,
            "chunk_size": DEFAULT_CHUNK_SIZE,
            "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
            "used_ocr": used_ocr,
            "exact_match_chunk_ids": retrieval.exact_match_chunk_ids,
            "adjacent_chunk_ids": retrieval.adjacent_chunk_ids,
            "sent": [
                {
                    "chunk_id": chunk.chunk_id,
                    "page": chunk.page,
                    "distance": chunk.distance,
                    "reason": chunk.reason,
                }
                for chunk in retrieval.selected
            ],
        },
        "result": result.model_dump(mode="json"),
    }


def prediction(record: dict, field: str) -> str | None:
    result = PaperResult.model_validate(record["result"])
    if result.classification is None:
        return None
    if field == "overall":
        return result.classification.overall
    return getattr(result.classification, field)


def write_report(output: Path, *, model: str, mode: RetrievalMode, top_k: int) -> None:
    latest = {}
    for line in output.read_text().splitlines():
        record = json.loads(line)
        if (
            record["result"]["provenance"]["prompt_sha256"] != IRIS_PROMPT_SHA256
            or record["result"]["provenance"]["model"] != model
            or record["retrieval"]["mode"] != mode
            or record["retrieval"]["top_k"] != top_k
        ):
            continue
        latest[record["evaluation_key"]] = record

    lines = [
        "# Structured top-20 classification evaluation",
        "",
        f"- Results: {len(latest)} reviewed papers",
        "- Baseline reviewed overall: 5 correct of 9 available, with 4 missing",
        f"- Model: `{model}`",
        f"- Prompt SHA-256: `{IRIS_PROMPT_SHA256}`",
        "",
        "| Field | Correct | Incorrect | Failed |",
        "|---|---:|---:|---:|",
    ]
    fields = ("overall", "observational_use", "synthetic_use", "synthetic_connection", "review_only")
    for field in fields:
        correct = incorrect = failed = 0
        for record in latest.values():
            actual = prediction(record, field)
            if actual is None:
                failed += 1
            elif actual == record["expected"][field]:
                correct += 1
            else:
                incorrect += 1
        lines.append(f"| `{field}` | {correct} | {incorrect} | {failed} |")

    lines.extend(
        [
            "",
            "| Paper | Expected | Predicted | Status |",
            "|---|---|---|---|",
        ]
    )
    for record in latest.values():
        predicted = prediction(record, "overall") or "-"
        lines.append(
            f"| `{record['id']}` | {record['expected']['overall']} | {predicted} | {record['result']['status']} |"
        )
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--retrieval-mode", choices=RetrievalMode, default=RetrievalMode.AUTO, type=RetrievalMode)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    cases = load_cases()
    if args.case_id:
        cases = [case for case in cases if case["id"] in args.case_id]
    if args.limit is not None:
        cases = cases[: args.limit]
    successful = set() if args.force else load_successful_keys(args.output)
    for index, case in enumerate(cases, start=1):
        key = evaluation_key(case, model=args.model, mode=args.retrieval_mode, top_k=args.top_k)
        if key in successful:
            print(f"[{index}/{len(cases)}] skip {case['id']}")
            continue
        print(f"[{index}/{len(cases)}] classify {case['id']}")
        record = classify_case(case, model=args.model, mode=args.retrieval_mode, top_k=args.top_k)
        append_record(args.output, record)
        print(f"[{index}/{len(cases)}] {record['result']['status']}")
    write_report(args.output, model=args.model, mode=args.retrieval_mode, top_k=args.top_k)


if __name__ == "__main__":
    main()
