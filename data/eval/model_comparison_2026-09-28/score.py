"""Reproduce the archived comparison offline; consensus agreement is not independent accuracy."""

import json
from pathlib import Path

ROOT = Path(__file__).parent


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score(predictions: dict, references: dict, ids: set[str]) -> str:
    errors = sum(predictions[paper] != references[paper]["include"] for paper in ids)
    return f"{errors}/{len(ids)}"


def main() -> None:
    references = {row["id"]: row for row in read_rows(ROOT / "reference_labels.jsonl")}
    historical = {paper for paper, row in references.items() if row["include"] is not None}
    usable = {paper for paper in historical if "exclude_reason" not in references[paper]}
    adjudicated = {paper for paper in usable if references[paper]["source"] == "adjudicated"}
    gold = {job["record_id"] for job in read_rows(ROOT / "jobs.jsonl") if job["expected"]["set"] == "gold"} & usable
    print("Errors / scored papers; the historical and usable totals include model-consensus labels.\n")
    print("| Model | Prompt | Historical | Usable | Adjudicated | Original gold | Failures |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for path in sorted(ROOT.glob("*.v4.*.jsonl")):
        rows = read_rows(path)
        predictions = {row["id"]: (row["result"]["classification"] or {}).get("include", "FAILED") for row in rows}
        if len(rows) != len(references) or predictions.keys() != references.keys():
            msg = f"Incomplete or duplicated benchmark records: {path.name}"
            raise ValueError(msg)
        model, version = path.stem.rsplit(".v", 1)
        scores = " | ".join(score(predictions, references, ids) for ids in (historical, usable, adjudicated, gold))
        failures = sum(value == "FAILED" for value in predictions.values())
        print(f"| {model} | {version} | {scores} | {failures} |")


if __name__ == "__main__":
    main()
