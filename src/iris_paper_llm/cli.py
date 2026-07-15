"""Command-line workflow for finding and classifying IRIS papers."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

from iris_paper_llm.ads import iris_query, search_papers
from iris_paper_llm.classify import DEFAULT_MODEL, classify_paths
from iris_paper_llm.download import (
    DEFAULT_BROWSER_WAIT_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    download_records,
)
from iris_paper_llm.evaluate import (
    DEFAULT_CASES,
    DEFAULT_OUTPUT,
    classify_cases,
    prepare_case_pdfs,
    write_report,
)
from iris_paper_llm.jsonl import read_jsonl
from iris_paper_llm.models import RetrievalMode
from iris_paper_llm.retrieval import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DEFAULT_TOP_K


def _add_classification_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--retrieval-mode",
        choices=RetrievalMode,
        default=RetrievalMode.AUTO,
        type=RetrievalMode,
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--no-ocr", action="store_true")


def _search(args: argparse.Namespace) -> dict[str, int]:
    query = args.query or iris_query(args.year)
    if args.output is None:
        if args.year is None:
            msg = "--output is required when using --query"
            raise ValueError(msg)
        args.output = Path("data/metadata") / f"{args.year}.jsonl"
    return search_papers(
        query,
        args.output,
        api_token=args.api_token,
        limit=args.limit,
        force=args.force,
    )


def _download(args: argparse.Namespace) -> dict[str, int]:
    records = read_jsonl(args.input)
    if args.limit is not None:
        records = records[: args.limit]
    output_dir = args.output_dir or Path("data/pdfs") / args.input.stem
    return download_records(
        records,
        output_dir,
        timeout=args.timeout,
        browser_fallback=args.browser_fallback,
        browser_wait=args.browser_wait,
    )


def _classify(args: argparse.Namespace) -> dict[str, int]:
    output = args.output or Path("data/results") / f"{args.input.name}.jsonl"
    summary = classify_paths(
        args.input,
        output,
        limit=args.limit,
        force=args.force,
        model=args.model,
        retrieval_mode=args.retrieval_mode,
        top_k=args.top_k,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        ocr_fallback=not args.no_ocr,
    )
    write_report(
        output,
        model=args.model,
        retrieval_mode=args.retrieval_mode,
        top_k=args.top_k,
    )
    return summary


def _evaluate(args: argparse.Namespace) -> dict[str, int]:
    preparation = {"downloaded": 0, "manual_queue": 0}
    if args.prepare_pdfs:
        preparation = prepare_case_pdfs(
            args.cases,
            timeout=args.timeout,
            browser_fallback=args.browser_fallback,
        )
    summary = classify_cases(
        args.cases,
        args.output,
        case_ids=args.case_id,
        limit=args.limit,
        force=args.force,
        model=args.model,
        retrieval_mode=args.retrieval_mode,
        top_k=args.top_k,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        ocr_fallback=not args.no_ocr,
    )
    write_report(
        args.output,
        model=args.model,
        retrieval_mode=args.retrieval_mode,
        top_k=args.top_k,
    )
    return {
        **summary,
        "downloaded": preparation["downloaded"],
        "manual_downloads": preparation["manual_queue"],
    }


def _run(args: argparse.Namespace) -> dict[str, int]:
    metadata = Path("data/metadata") / f"{args.year}.jsonl"
    pdfs = Path("data/pdfs") / str(args.year)
    results = Path("data/results") / f"{args.year}.jsonl"
    search = search_papers(
        args.query or iris_query(args.year),
        metadata,
        api_token=args.api_token,
        limit=args.limit,
        force=args.force,
    )
    records = read_jsonl(metadata)[: args.limit]
    downloads = download_records(
        records,
        pdfs,
        timeout=args.timeout,
        browser_fallback=args.browser_fallback,
        browser_wait=args.browser_wait,
    )
    classifications = classify_paths(
        pdfs,
        results,
        limit=args.limit,
        force=args.force,
        model=args.model,
        retrieval_mode=args.retrieval_mode,
        top_k=args.top_k,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        ocr_fallback=not args.no_ocr,
    )
    write_report(
        results,
        model=args.model,
        retrieval_mode=args.retrieval_mode,
        top_k=args.top_k,
    )
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
    search.add_argument("--api-token")
    search.add_argument("--limit", type=int, default=2000)
    search.add_argument("--force", action="store_true")
    search.set_defaults(handler=_search)

    download = subparsers.add_parser("download", help="Download and validate PDFs from metadata JSONL.")
    download.add_argument("input", type=Path)
    download.add_argument("--output-dir", type=Path)
    download.add_argument("--limit", type=int)
    download.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    download.add_argument("--browser-fallback", action="store_true")
    download.add_argument("--browser-wait", type=float, default=DEFAULT_BROWSER_WAIT_SECONDS)
    download.set_defaults(handler=_download)

    classify = subparsers.add_parser("classify", help="Classify one PDF or a directory sequentially.")
    classify.add_argument("input", type=Path)
    classify.add_argument("--output", type=Path)
    classify.add_argument("--limit", type=int)
    classify.add_argument("--force", action="store_true")
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
    evaluate.add_argument("--browser-fallback", action="store_true")
    _add_classification_options(evaluate)
    evaluate.set_defaults(handler=_evaluate)

    run = subparsers.add_parser("run", help="Compose search, download, classification, and reporting.")
    run.add_argument("--year", required=True, type=int)
    run.add_argument("--query", help="Override the standard IRIS query for this year.")
    run.add_argument("--api-token")
    run.add_argument("--limit", type=int, default=2000)
    run.add_argument("--force", action="store_true")
    run.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    run.add_argument("--browser-fallback", action="store_true")
    run.add_argument("--browser-wait", type=float, default=DEFAULT_BROWSER_WAIT_SECONDS)
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
    except (FileNotFoundError, ValueError, requests.RequestException) as error:
        parser.error(str(error))
    print(json.dumps(summary, sort_keys=True))
