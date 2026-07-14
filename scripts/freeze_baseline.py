"""Freeze the existing 2025 v2 run without calling ADS or OpenAI."""

# ruff: noqa: EM101, EM102, T201, TRY003

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "eval" / "phase0_v2_2025"

LIBRARY_FILE = ROOT / "data" / "bibcodes" / "iris_library_bibcodes.txt"
CANDIDATE_FILE = ROOT / "data" / "bibcodes" / "iris_search_bibcodes.txt"
FAILED_FILE = ROOT / "data" / "pdfs" / "iris_search" / "failed_bibcodes.txt"
PDF_DIR = ROOT / "data" / "pdfs" / "iris_search"
RESULTS_FILE = ROOT / "scripts" / "iris_search_results.json"
REVIEWED_CASES_FILE = ROOT / "data" / "eval" / "reviewed_cases.jsonl"
CONFIG_FILE = ROOT / "src" / "paper_data_linking" / "web_app" / "config" / "iris_config_v2.yaml"
MODEL_DIR = ROOT / "models" / "onnx"

LIBRARY_SNAPSHOT_DATE = "2026-01-07"
BASELINE_CREATED_ON = "2026-07-14"
SEARCH_QUERY = (
    '=full:"Interface Region Imaging Spectrograph" OR '
    "citations(bibcode:2014SoPh..289.2733D) + property:refereed + "
    'doctype:"Article" + pubdate:[2025-01 TO 2025-12]'
)
VALID_LABELS = {"YES", "NO", "UNCERTAIN"}

IMPLEMENTATION_FILES = (
    ROOT / "Makefile",
    ROOT / "scripts" / "query_api.py",
    ROOT / "src" / "paper_data_linking" / "process" / "analyzer.py",
    ROOT / "src" / "paper_data_linking" / "process" / "embedders.py",
    ROOT / "src" / "paper_data_linking" / "process" / "plugins.py",
    ROOT / "src" / "paper_data_linking" / "process" / "splitters.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def source_record(path: Path) -> dict[str, str]:
    return {"path": relative(path), "sha256": sha256_file(path)}


def load_bibcodes(path: Path) -> list[str]:
    bibcodes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(bibcodes) != len(set(bibcodes)):
        raise ValueError(f"Duplicate bibcodes in {relative(path)}")
    return bibcodes


def load_results() -> dict[str, dict]:
    results = json.loads(RESULTS_FILE.read_text())
    if not isinstance(results, dict):
        raise TypeError(f"Expected an object in {relative(RESULTS_FILE)}")
    for bibcode, result in results.items():
        if not isinstance(result, dict):
            raise TypeError(f"Result for {bibcode} is not an object")
        normalize_label(result.get("iris_classification"))
    return results


def load_reviewed_cases() -> list[dict]:
    cases = [json.loads(line) for line in REVIEWED_CASES_FILE.read_text().splitlines() if line.strip()]
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs in reviewed_cases.jsonl")

    for case in cases:
        status = case["review_status"]
        if status not in {"RESOLVED", "NEEDS_SCOPE_REVIEW"}:
            raise ValueError(f"Invalid review status for {case['id']}: {status}")
        if status == "RESOLVED":
            for key in (
                "expected_overall",
                "expected_observational_use",
                "expected_synthetic_use",
                "expected_review_only",
            ):
                normalize_label(case[key])
            expected = (
                "YES"
                if "YES"
                in {
                    case["expected_observational_use"],
                    case["expected_synthetic_use"],
                }
                else "NO"
            )
            if case["expected_overall"] != expected:
                raise ValueError(f"Inconsistent overall label for {case['id']}")
            connection = case["synthetic_iris_connection"]
            allowed = {"EXPLICIT", "PASSBAND_ONLY"} if case["expected_synthetic_use"] == "YES" else {"NOT_APPLICABLE"}
            if connection not in allowed:
                raise ValueError(f"Inconsistent synthetic connection for {case['id']}")

        pdf_path = ROOT / case["path"]
        if pdf_path.exists() and sha256_file(pdf_path) != case["pdf_sha256"]:
            raise ValueError(f"PDF checksum changed for {case['id']}")
    return cases


def normalize_label(value: object) -> str:
    if not isinstance(value, str) or value.strip().upper() not in VALID_LABELS:
        raise ValueError(f"Invalid classification: {value!r}")
    return value.strip().upper()


def summarize_reviewed(cases: list[dict], results: dict[str, dict]) -> tuple[dict, list[str]]:
    resolved = [case for case in cases if case["review_status"] == "RESOLVED"]
    pending = [case["id"] for case in cases if case["review_status"] != "RESOLVED"]
    categories = {
        "observational": lambda case: case["expected_observational_use"] == "YES",
        "synthetic": lambda case: case["expected_synthetic_use"] == "YES",
        "review_only": lambda case: case["expected_review_only"] == "YES",
        "overall": lambda _case: True,
    }
    summaries = {}
    for name, include in categories.items():
        selected = [case for case in resolved if include(case)]
        counts = Counter()
        for case in selected:
            result = results.get(case.get("bibcode"))
            if result is None:
                counts["missing_baseline"] += 1
                continue
            prediction = normalize_label(result["iris_classification"])
            counts["correct" if prediction == case["expected_overall"] else "incorrect"] += 1
        summaries[name] = {
            "cases": len(selected),
            "with_baseline": len(selected) - counts["missing_baseline"],
            "correct": counts["correct"],
            "incorrect": counts["incorrect"],
            "missing_baseline": counts["missing_baseline"],
        }
    return summaries, pending


def build_outputs() -> dict[Path, str]:
    library = load_bibcodes(LIBRARY_FILE)
    candidates = load_bibcodes(CANDIDATE_FILE)
    failed = set(load_bibcodes(FAILED_FILE))
    results = load_results()
    reviewed_cases = load_reviewed_cases()

    library_set = set(library)
    candidate_set = set(candidates)
    result_set = set(results)
    if not result_set <= candidate_set:
        raise ValueError(f"Results outside candidate set: {sorted(result_set - candidate_set)}")
    if not failed <= candidate_set:
        raise ValueError(f"Failures outside candidate set: {sorted(failed - candidate_set)}")

    records = []
    for bibcode in candidates:
        result = results.get(bibcode)
        pdf_path = PDF_DIR / f"{bibcode}.pdf"
        status = "CLASSIFIED" if result is not None else "DOWNLOAD_FAILED" if bibcode in failed else "MISSING_RESULT"
        records.append(
            {
                "schema_version": 1,
                "bibcode": bibcode,
                "ground_truth": "YES" if bibcode in library_set else "UNLABELED",
                "status": status,
                "pdf_sha256": sha256_file(pdf_path) if pdf_path.exists() else None,
                "saved_v2_result": result,
            }
        )

    predictions = Counter(normalize_label(result["iris_classification"]) for result in results.values())
    library_scope = library_set & candidate_set
    library_evaluation = Counter()
    for bibcode in library_scope:
        result = results.get(bibcode)
        if result is None:
            library_evaluation["pipeline_failure"] += 1
            continue
        prediction = normalize_label(result["iris_classification"])
        outcome = {"YES": "recovered_positive", "NO": "false_negative", "UNCERTAIN": "unresolved_positive"}[prediction]
        library_evaluation[outcome] += 1

    outside_library_yes = sorted(
        bibcode
        for bibcode, result in results.items()
        if bibcode not in library_set and normalize_label(result["iris_classification"]) == "YES"
    )
    reviewed_negative_bibcodes = {
        case["bibcode"]
        for case in reviewed_cases
        if case["review_status"] == "RESOLVED" and case["expected_overall"] == "NO" and case.get("bibcode")
    }
    outside_library_reviewed_negative = sorted(set(outside_library_yes) & reviewed_negative_bibcodes)
    outside_library_review_queue = sorted(set(outside_library_yes) - reviewed_negative_bibcodes)
    reviewed_summary, pending_review = summarize_reviewed(reviewed_cases, results)

    model_files = [source_record(path) for path in sorted(MODEL_DIR.rglob("*")) if path.is_file()]
    manifest = {
        "schema_version": 1,
        "baseline_id": "phase0_v2_2025",
        "created_on": BASELINE_CREATED_ON,
        "scope": {
            "search_query": SEARCH_QUERY,
            "candidate_period": "2025-01 through 2025-12",
            "ads_library_id": "30bDOCvOTJiAgacWhJxkmA",
            "ads_library_snapshot_date": LIBRARY_SNAPSHOT_DATE,
            "ground_truth_rule": "ADS library membership is positive; absence is unlabeled.",
        },
        "configuration": {
            "config": source_record(CONFIG_FILE),
            "model_name": "gpt-5-mini",
            "model_snapshot_pinned": False,
            "temperature": 0,
            "splitter": "PyMuPDFTokenSplitter",
            "chunk_size_tokens": 500,
            "chunk_overlap_tokens": 50,
            "heuristic_threshold": 80,
            "retrieval_count": 10,
            "embedder": "Chroma DefaultEmbeddingFunction (all-MiniLM-L6-v2 ONNX)",
            "embedder_filter": {"passed_iris_heuristic": 1},
            "onnx_files": model_files,
            "implementation_files": [source_record(path) for path in IMPLEMENTATION_FILES],
        },
        "source_artifacts": [
            source_record(LIBRARY_FILE),
            source_record(CANDIDATE_FILE),
            source_record(FAILED_FILE),
            source_record(RESULTS_FILE),
            source_record(REVIEWED_CASES_FILE),
            source_record(Path(__file__).resolve()),
        ],
        "counts": {
            "library_snapshot": len(library),
            "candidates": len(candidates),
            "classified": len(results),
            "missing_results": len(candidate_set - result_set),
            "download_failures": len(failed),
            "predictions": {label: predictions[label] for label in sorted(VALID_LABELS)},
            "library_in_candidate_scope": len(library_scope),
            "library_evaluation": {
                key: library_evaluation[key]
                for key in ("recovered_positive", "false_negative", "unresolved_positive", "pipeline_failure")
            },
            "outside_library_predicted_yes": len(outside_library_yes),
            "outside_library_reviewed_negative": len(outside_library_reviewed_negative),
            "outside_library_pending_review": len(outside_library_review_queue),
        },
        "outside_library_reviewed_negative": outside_library_reviewed_negative,
        "outside_library_review_queue": outside_library_review_queue,
        "reviewed_case_evaluation": reviewed_summary,
        "pending_scope_review": pending_review,
        "known_limitations": [
            "The saved result writer discarded all but the first aspect, and the v2 parser failed to preserve aspects.",
            "Selected chunk IDs, similarity scores, token usage, and request IDs were not persisted and cannot be reconstructed from saved output.",
            "Existing results did not persist per-paper prompt/model provenance; the v2 config is the reported run configuration, not cryptographically linked to each response.",
            "gpt-5-mini was recorded as a moving alias rather than a pinned model snapshot.",
            "The ADS library supplies positive labels only; candidates outside it are unlabeled until reviewed.",
            "No genuinely uncertain paper was found in the local corpus; full-text review resolved every candidate considered.",
        ],
    }

    records_text = "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records)
    library_text = "\n".join(library) + "\n"
    candidates_text = "\n".join(candidates) + "\n"
    review_queue_text = "\n".join(outside_library_review_queue) + "\n"
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    report_text = build_report(manifest)

    return {
        OUTPUT_DIR / f"ads_iris_library_{LIBRARY_SNAPSHOT_DATE}.txt": library_text,
        OUTPUT_DIR / "search_candidates_2025.txt": candidates_text,
        OUTPUT_DIR / "baseline_results.jsonl": records_text,
        OUTPUT_DIR / "outside_library_review_queue.txt": review_queue_text,
        OUTPUT_DIR / "manifest.json": manifest_text,
        OUTPUT_DIR / "report.md": report_text,
    }


def build_report(manifest: dict) -> str:
    counts = manifest["counts"]
    library = counts["library_evaluation"]
    reviewed = manifest["reviewed_case_evaluation"]
    rows = "\n".join(
        f"| {name} | {values['cases']} | {values['with_baseline']} | {values['correct']} | "
        f"{values['incorrect']} | {values['missing_baseline']} |"
        for name, values in reviewed.items()
    )
    pending = ", ".join(f"`{case_id}`" for case_id in manifest["pending_scope_review"]) or "None"
    limitations = "\n".join(f"- {item}" for item in manifest["known_limitations"])
    return f"""# Phase 0 v2 baseline report

Created {manifest["created_on"]} from the saved 2025 run. No ADS or OpenAI
request was made while creating this snapshot.

## Corpus

- ADS library snapshot: {counts["library_snapshot"]} positive bibcodes
- 2025 search candidates: {counts["candidates"]}
- Saved classifications: {counts["classified"]}
- Missing results/download failures: {counts["missing_results"]}/{counts["download_failures"]}
- Predictions: {counts["predictions"]["YES"]} YES, {counts["predictions"]["NO"]} NO,
  {counts["predictions"]["UNCERTAIN"]} UNCERTAIN

## Curated ADS-library positives in candidate scope

| Outcome | Count |
| --- | ---: |
| Expected positives | {counts["library_in_candidate_scope"]} |
| Recovered positive | {library["recovered_positive"]} |
| False negative | {library["false_negative"]} |
| Unresolved positive | {library["unresolved_positive"]} |
| Pipeline failure | {library["pipeline_failure"]} |

There were {counts["outside_library_predicted_yes"]} YES predictions outside
the ADS library. {counts["outside_library_reviewed_negative"]} is now a
reviewed negative, while {counts["outside_library_pending_review"]} remain
queued and are not counted as false positives.

## Manually reviewed cases

These rows compare the saved *overall* v2 prediction within each scientific
case category. Component-level v2 predictions were not stored reliably.

| Category | Cases | With baseline | Correct | Incorrect | Missing baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
{rows}

Pending scope review: {pending}

## Known limitations

{limitations}
"""


def write_outputs(outputs: dict[Path, str]) -> None:
    for path, text in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)


def check_outputs(outputs: dict[Path, str]) -> None:
    mismatches = [
        relative(path) for path, expected in outputs.items() if not path.exists() or path.read_text() != expected
    ]
    if mismatches:
        raise SystemExit("Baseline artifacts differ: " + ", ".join(mismatches))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify tracked artifacts without rewriting them.")
    args = parser.parse_args()
    outputs = build_outputs()
    if args.check:
        check_outputs(outputs)
        print("Phase 0 baseline artifacts are current.")
    else:
        write_outputs(outputs)
        print(f"Wrote {len(outputs)} baseline artifacts to {relative(OUTPUT_DIR)}.")


if __name__ == "__main__":
    main()
