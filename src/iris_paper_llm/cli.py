"""Command-line workflow for finding and classifying IRIS papers."""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import openai
import requests
from dotenv import load_dotenv

from iris_paper_llm.ads import fetch_library, iris_query, search_papers
from iris_paper_llm.classify import (
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    OPENAI_MAX_RETRIES,
    OPENAI_TIMEOUT_SECONDS,
    classify_jobs,
    pdf_jobs,
)
from iris_paper_llm.download import DEFAULT_TIMEOUT_SECONDS, download_records
from iris_paper_llm.evaluate import (
    DEFAULT_CASES,
    DEFAULT_OUTPUT,
    case_jobs,
    prepare_case_pdfs,
    write_report,
)
from iris_paper_llm.jsonl import read_jsonl


def _add_classification_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)


def _openai_client(model: str) -> openai.OpenAI:
    """Check the OpenAI key and model with a free request, so a bad setup fails before any search or download."""
    client = openai.OpenAI(timeout=OPENAI_TIMEOUT_SECONDS, max_retries=OPENAI_MAX_RETRIES)
    client.models.retrieve(model)
    return client


def _search(args: argparse.Namespace) -> dict[str, int]:
    query = args.query or iris_query(args.year)
    if args.output is None:
        if args.year is None:
            msg = "--output is required when using --query"
            raise ValueError(msg)
        args.output = Path("data/metadata") / f"{args.year}.jsonl"
    return search_papers(query, args.output, limit=args.limit)


def _library(args: argparse.Namespace) -> Path:
    """The --library file, or else today's ADS IRIS library, fetched to data/."""
    if args.library is None:
        args.library = Path("data") / f"ads_iris_library_{datetime.now().astimezone().date()}.txt"
        fetch_library(args.library)
    return args.library


def _download(args: argparse.Namespace) -> dict[str, int]:
    records = read_jsonl(args.input)
    if args.limit is not None:
        records = records[: args.limit]
    output_dir = args.output_dir or Path("data/pdfs") / args.input.stem
    return download_records(records, output_dir, timeout=args.timeout)


def _classify(args: argparse.Namespace) -> dict[str, int]:
    output = args.output or Path("data/results") / f"{args.input.name}.jsonl"
    client = _openai_client(args.model)
    library = _library(args)
    summary = classify_jobs(
        pdf_jobs(args.input, args.limit),
        output,
        force=args.force,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        client=client,
    )
    write_report(output, library=library, model=args.model, reasoning_effort=args.reasoning_effort)
    return summary


def _evaluate(args: argparse.Namespace) -> dict[str, int]:
    client = _openai_client(args.model)
    preparation = {"downloaded": 0, "manual_queue": 0}
    if args.prepare_pdfs:
        preparation = prepare_case_pdfs(args.cases, timeout=args.timeout)
    summary = classify_jobs(
        case_jobs(args.cases, case_ids=args.case_id, limit=args.limit),
        args.output,
        force=args.force,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        client=client,
    )
    write_report(args.output, cases=args.cases, model=args.model, reasoning_effort=args.reasoning_effort)
    return {
        **summary,
        "downloaded": preparation["downloaded"],
        "manual_downloads": preparation["manual_queue"],
    }


def _run(args: argparse.Namespace) -> dict[str, int]:
    metadata = Path("data/metadata") / f"{args.year}.jsonl"
    pdfs = Path("data/pdfs") / str(args.year)
    results = Path("data/results") / f"{args.year}.jsonl"
    client = _openai_client(args.model)
    library = _library(args)
    search = search_papers(args.query or iris_query(args.year), metadata, limit=args.limit)
    downloads = download_records(read_jsonl(metadata), pdfs, timeout=args.timeout)
    classifications = classify_jobs(
        pdf_jobs(pdfs, args.limit),
        results,
        force=args.force,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        client=client,
    )
    write_report(results, library=library, model=args.model, reasoning_effort=args.reasoning_effort)
    return {
        **classifications,
        "downloaded": downloads["downloaded"],
        "manual_downloads": downloads["manual_queue"],
        "metadata_found": search["found"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="iris-papers", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    search = subparsers.add_parser("search", help="Search ADS and write download metadata.")
    source = search.add_mutually_exclusive_group(required=True)
    source.add_argument("--year", type=int)
    source.add_argument("--query")
    search.add_argument("--output", type=Path)
    search.add_argument("--limit", type=int, default=2000)
    search.set_defaults(handler=_search)

    download = subparsers.add_parser("download", help="Download and validate PDFs from metadata JSONL.")
    download.add_argument("input", type=Path)
    download.add_argument("--output-dir", type=Path)
    download.add_argument("--limit", type=int)
    download.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    download.set_defaults(handler=_download)

    classify = subparsers.add_parser("classify", help="Classify one PDF or a directory sequentially.")
    classify.add_argument("input", type=Path)
    classify.add_argument("--output", type=Path)
    classify.add_argument("--limit", type=int)
    classify.add_argument("--force", action="store_true")
    classify.add_argument(
        "--library", type=Path, help="ADS IRIS library bibcodes, one per line (default: fetch the live library)."
    )
    _add_classification_options(classify)
    classify.set_defaults(handler=_classify)

    evaluate = subparsers.add_parser("evaluate", help="Run the checksum-frozen reviewed corpus.")
    evaluate.add_argument("cases", nargs="?", type=Path, default=DEFAULT_CASES)
    evaluate.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    evaluate.add_argument("--case-id", action="append")
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--force", action="store_true")
    evaluate.add_argument("--prepare-pdfs", action="store_true")
    evaluate.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    _add_classification_options(evaluate)
    evaluate.set_defaults(handler=_evaluate)

    run = subparsers.add_parser("run", help="Compose search, download, classification, and reporting.")
    run.add_argument("--year", required=True, type=int)
    run.add_argument("--query", help="Override the standard IRIS query for this year.")
    run.add_argument("--limit", type=int, default=2000)
    run.add_argument("--force", action="store_true", help="Reclassify papers that already have a result.")
    run.add_argument(
        "--library", type=Path, help="ADS IRIS library bibcodes, one per line (default: fetch the live library)."
    )
    run.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    _add_classification_options(run)
    run.set_defaults(handler=_run)
    return parser


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = args.handler(args)
    except (FileNotFoundError, ValueError, requests.RequestException, openai.OpenAIError) as error:
        parser.error(str(error))
    print(json.dumps(summary, sort_keys=True))
