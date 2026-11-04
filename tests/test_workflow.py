from __future__ import annotations

import hashlib
import json
import re
import shutil
from contextlib import chdir
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import openai
import pytest

from iris_paper_llm.ads import fetch_library, iris_query, search_papers
from iris_paper_llm.classify import (
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    classification_key,
    classify_jobs,
    pdf_jobs,
)
from iris_paper_llm.cli import build_parser
from iris_paper_llm.evaluate import write_report
from iris_paper_llm.jsonl import append_jsonl, read_jsonl, write_jsonl
from iris_paper_llm.models import (
    Basis,
    Decision,
    Evidence,
    IRISClassification,
    PaperResult,
    PipelineProvenance,
    ResultStatus,
)


def observational(page: int) -> IRISClassification:
    return IRISClassification(
        include=Decision.YES,
        basis=[Basis.OBSERVATIONAL_DATA],
        iris_mission_mentioned=True,
        evidence=[Evidence(page=page, reason="OBSERVATIONAL_DATA: IRIS data are analyzed.")],
    )


class FakeResponses:
    def parse(self, *, input: list[dict], **_kwargs: object) -> object:  # noqa: A002  (the OpenAI keyword)
        context = input[1]["content"]
        page = int(context.split("[PAGE ", 1)[1].split("]", 1)[0])
        return SimpleNamespace(
            id="response-test",
            model="resolved-test-model",
            output=[],
            output_parsed=observational(page),
            status="completed",
            usage=None,
        )


class FailingResponses:
    @staticmethod
    def parse(**_kwargs: object) -> object:
        msg = "offline API failure"
        raise RuntimeError(msg)


def test_offline_pipeline_checkpoint_and_resume() -> None:
    client = SimpleNamespace(responses=FakeResponses())
    repository = Path(__file__).resolve().parents[1]
    first = repository / "tests/data/pdfs/observational_filament_flows.pdf"
    second = repository / "tests/data/pdfs/iris_seen_but_not_used.pdf"
    with TemporaryDirectory() as directory:
        root = Path(directory)
        pdfs = root / "pdfs"
        pdfs.mkdir()
        shutil.copyfile(first, pdfs / first.name)
        shutil.copyfile(second, pdfs / second.name)
        output = root / "results.jsonl"

        first_summary = classify_jobs(pdf_jobs(first), output, client=client)
        assert len(output.read_text().splitlines()) == 1
        directory_summary = classify_jobs(pdf_jobs(pdfs), output, client=client)

        failed_output = root / "failed.jsonl"
        failed_summary = classify_jobs(
            pdf_jobs(first), failed_output, client=SimpleNamespace(responses=FailingResponses())
        )

        assert first_summary == {"positive": 1, "negative": 0, "uncertain": 0, "failed": 0, "skipped": 0}
        assert directory_summary == {"positive": 1, "negative": 0, "uncertain": 0, "failed": 0, "skipped": 1}
        records = read_jsonl(output)
        assert len(records) == 2
        assert set(records[0]) == {"evaluation_key", "id", "bibcode", "pages_sent", "characters_sent", "result"}
        assert records[0]["pages_sent"] > 1
        assert records[0]["characters_sent"] > 10_000
        result = PaperResult.model_validate(records[0]["result"])
        assert result.status == ResultStatus.CLASSIFIED
        assert result.classification is not None
        assert result.classification.include == Decision.YES
        assert result.classification.evidence[0].page == 1

        assert failed_summary == {"positive": 0, "negative": 0, "uncertain": 0, "failed": 1, "skipped": 0}
        failed = PaperResult.model_validate(read_jsonl(failed_output)[0]["result"])
        assert failed.status == ResultStatus.PROCESSING_FAILED
        assert failed.errors == ["offline API failure"]
        report = write_report(output)
        assert report["positive"] == 2
        assert output.with_suffix(".md").is_file()


def test_july_rerun_after_ads_replaced_a_bibcode() -> None:
    """In January a provisional bibcode was classified; in July ADS returns its final one, which the library has."""
    client = SimpleNamespace(responses=FakeResponses())
    pdf = Path(__file__).resolve().parent / "data/pdfs/observational_filament_flows.pdf"
    old, new = "2025MNRAS.tmp..123X", "2025MNRAS.540..123X"
    with TemporaryDirectory() as directory:
        root = Path(directory)
        pdfs = root / "pdfs"
        pdfs.mkdir()
        output = root / "results.jsonl"
        shutil.copyfile(pdf, pdfs / f"{old}.pdf")
        classify_jobs(pdf_jobs(pdfs), output, client=client)
        shutil.copyfile(pdf, pdfs / f"{new}.pdf")  # the same bytes, downloaded again under the final bibcode
        july = classify_jobs(pdf_jobs(pdfs), output, client=client)
        library = root / "library.txt"
        library.write_text(f"{new}\n")
        write_report(output, library=library, pdfs=pdfs, papers={new})
        report = output.with_suffix(".md").read_text()
        review = (root / "results_to_review.txt").read_text()
    assert july == {"positive": 1, "negative": 0, "uncertain": 0, "failed": 0, "skipped": 1}
    assert "- Results: 1 papers" in report
    assert old not in report
    assert review == ""


def test_classify_reports_the_current_search_only_for_its_own_directory() -> None:
    client = SimpleNamespace(responses=FakeResponses())
    pdf = Path(__file__).resolve().parent / "data/pdfs/observational_filament_flows.pdf"
    old, new = "2025MNRAS.tmp..123X", "2025MNRAS.540..123X"
    reports = {}
    with (
        TemporaryDirectory() as directory,
        chdir(directory),
        patch("iris_paper_llm.cli._openai_client", return_value=client),
    ):
        write_jsonl(Path("data/metadata/2025.jsonl"), [{"bibcode": new}])
        Path("library.txt").write_text("")
        for pdfs in (Path("data/pdfs/2025"), Path("elsewhere/2025")):  # the second is not the search's directory
            pdfs.mkdir(parents=True)
            for bibcode in (old, new):
                shutil.copyfile(pdf, pdfs / f"{bibcode}.pdf")
            args = build_parser().parse_args(["classify", str(pdfs), "--library", "library.txt", "--output", "r.jsonl"])
            args.handler(args)
            reports[pdfs.parts[0]] = Path("r.md").read_text()
    assert new in reports["data"]
    assert old not in reports["data"]
    assert old in reports["elsewhere"]


def result_record(paper: str, status: ResultStatus, *, model: str = DEFAULT_MODEL, pdf: str = "") -> dict:
    pdf_hash = hashlib.sha256((pdf or paper).encode()).hexdigest()
    classified = status == ResultStatus.CLASSIFIED
    result = PaperResult(
        bibcode=paper,
        pdf_sha256=pdf_hash,
        status=status,
        classification=observational(1) if classified else None,
        provenance=PipelineProvenance(
            pipeline_version="test",
            prompt_version="test",
            prompt_sha256="0" * 64,
            model=model,
            reasoning_effort=DEFAULT_REASONING_EFFORT,
            request_id=None,
            input_tokens=None,
            output_tokens=None,
        ),
        errors=[] if classified else ["offline API failure"],
    )
    return {
        "evaluation_key": classification_key(pdf_hash, model=model, reasoning_effort=DEFAULT_REASONING_EFFORT),
        "id": paper,
        "bibcode": paper,
        "pages_sent": 1,
        "characters_sent": 100,
        "result": result.model_dump(mode="json"),
    }


def test_report_prefers_latest_success_without_double_counting() -> None:
    with TemporaryDirectory() as directory:
        output = Path(directory) / "results.jsonl"
        for record in (
            result_record("success-then-failure", ResultStatus.CLASSIFIED),
            result_record("success-then-failure", ResultStatus.PROCESSING_FAILED),
            result_record("failure-then-success", ResultStatus.PROCESSING_FAILED),
            result_record("failure-then-success", ResultStatus.CLASSIFIED),
            result_record("never-classified", ResultStatus.PROCESSING_FAILED),
            result_record("never-classified", ResultStatus.PROCESSING_FAILED),
            result_record("other-model", ResultStatus.CLASSIFIED, model="other-model"),
            result_record("replaced-pdf", ResultStatus.CLASSIFIED),
            result_record("replaced-pdf", ResultStatus.PROCESSING_FAILED, pdf="publisher copy"),
            result_record("relabelled", ResultStatus.CLASSIFIED) | {"expected": {"include": "NO", "basis": []}},
        ):
            append_jsonl(output, record)
        cases = Path(directory) / "cases.jsonl"
        case = {"id": "relabelled", "expected_include": "YES", "expected_basis": ["OBSERVATIONAL_DATA"]}
        cases.write_text(json.dumps(case) + "\n")
        summary = write_report(output, cases=cases)
        report = output.with_suffix(".md").read_text()
    assert summary == {"positive": 3, "negative": 0, "uncertain": 0, "failed": 2, "skipped": 0}
    assert "- Results: 5 papers" in report
    assert report.count("`success-then-failure`") == 1
    assert "| `success-then-failure` | - | YES | OBSERVATIONAL_DATA | CLASSIFIED |" in report
    assert "| `replaced-pdf` | offline API failure |" in report
    assert "| `replaced-pdf` | - | - | - | PROCESSING_FAILED |" in report
    assert "other-model" not in report
    assert "| 1 | 0 | 0 | 1 of 1 |" in report
    assert "| `relabelled` | YES (OBSERVATIONAL_DATA) | YES |" in report


def test_report_lists_new_candidates_and_failures() -> None:
    with TemporaryDirectory() as directory:
        output = Path(directory) / "results.jsonl"
        failed = result_record("2025ApJ...3C", ResultStatus.PROCESSING_FAILED)
        failed["result"]["errors"] = ["1 validation error for IRISClassification\n  Value error | basis"]
        for record in (
            result_record("2025ApJ...1A", ResultStatus.CLASSIFIED),
            result_record("2025A&A...2B", ResultStatus.CLASSIFIED),
            failed,
        ):
            append_jsonl(output, record)
        library = Path(directory) / "library.txt"
        library.write_text("2025ApJ...1A\n")
        pdfs = Path(directory) / "pdfs"
        pdfs.mkdir()
        (pdfs / "2025ApJ...5E.pdf").write_bytes(b"%PDF-")  # saved by hand since the queue was written
        queue = [
            {
                "bibcode": "2025A&A...4D",
                "target": str(pdfs / "2025A&A...4D.pdf"),
                "category": "blocked",
                "errors": ["HTTP 403 (bot protection: DataDome)", "HTTP 403 (bot protection: DataDome)"],
            },
            {
                "bibcode": "2025ApJ...5E",
                "target": str(pdfs / "2025ApJ...5E.pdf"),
                "category": "permanent",
                "errors": [],
            },
        ]
        (pdfs / "manual_downloads.jsonl").write_text("".join(json.dumps(entry) + "\n" for entry in queue))
        write_report(output, library=library, pdfs=pdfs)
        report = output.with_suffix(".md").read_text()
        review = (Path(directory) / "results_to_review.txt").read_text().splitlines()
    assert review == ["https://ui.adsabs.harvard.edu/abs/2025A%26A...2B/abstract  # YES (OBSERVATIONAL_DATA)"]
    assert "not in `library.txt` (1)" in report
    assert "[2025ApJ...1A](" not in report
    assert "| `2025ApJ...3C` | 1 validation error for IRISClassification Value error / basis |" in report
    assert "## No PDF, not classified (1)" in report
    assert (
        "| [2025A&A...4D](https://ui.adsabs.harvard.edu/abs/2025A%26A...4D/abstract) "
        f"| blocked: HTTP 403 (bot protection: DataDome) | `{pdfs / '2025A&A...4D.pdf'}` |"
    ) in report
    assert "2025ApJ...5E" not in report


@pytest.mark.parametrize(("lost", "expected_starts"), [(0, [0, 2]), (1, [0, 2, 3])])
def test_library_fetch_paginates(lost: int, expected_starts: list[int]) -> None:
    """ADS counts `lost` more stored bibcodes than its index still finds and returns."""
    bibcodes = ["2014A", "2015B", "2016C"]
    starts = []

    def get(_url: str, *, params: dict, **_kwargs: object) -> SimpleNamespace:
        starts.append(params["start"])
        page = bibcodes[params["start"] : params["start"] + 2]  # the server caps a page at 2 rows
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"documents": page, "metadata": {"num_documents": len(bibcodes) + lost}},
        )

    with (
        TemporaryDirectory() as directory,
        patch("iris_paper_llm.ads.requests.get", side_effect=get),
        patch.dict("os.environ", {"ADS_TOKEN": "token"}),
    ):
        output = Path(directory) / "library.txt"
        assert fetch_library(output) == 3
        assert output.read_text().split() == bibcodes
    assert starts == expected_starts


def test_run_checks_the_openai_key_before_searching() -> None:
    args = build_parser().parse_args(["run", "--year", "2025"])
    with (
        patch.dict("os.environ", {"ADS_TOKEN": "token"}, clear=True),
        patch("iris_paper_llm.cli.fetch_library") as fetch,
        patch("iris_paper_llm.cli.search_papers") as search,
        pytest.raises(openai.OpenAIError),
    ):
        args.handler(args)
    fetch.assert_not_called()
    search.assert_not_called()


def test_standard_query_is_scoped_to_year() -> None:
    query = iris_query(2025)
    assert "citations(bibcode:2014SoPh..289.2733D)" in query
    assert query.endswith("pubdate:[2025-01 TO 2025-12]")


def test_ads_search_paginates_and_always_requeries() -> None:
    documents = [
        {"bibcode": "A", "links_data": ['{"access":"open","type":"pdf","url":"https://example.test/a.pdf"}']},
        {"bibcode": "B"},
        {"bibcode": "C", "links_data": []},
    ]
    calls = []

    def get(_url: str, *, params: dict, **_kwargs: object) -> SimpleNamespace:
        calls.append(params)
        start = params["start"]
        page = documents[start : start + min(params["rows"], 2)]  # the server caps a page at 2 rows
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"response": {"numFound": len(documents), "start": start, "docs": page}},
        )

    with (
        TemporaryDirectory() as directory,
        patch("iris_paper_llm.ads.requests.get", side_effect=get),
        patch.dict("os.environ", {"ADS_TOKEN": "token"}),
    ):
        output = Path(directory) / "metadata.jsonl"
        assert search_papers("q", output) == {"found": 3, "written": 3}
        assert search_papers("q", output) == {"found": 3, "written": 3}
        assert [call["start"] for call in calls] == [0, 2, 0, 2]
        records = read_jsonl(output)
        assert search_papers("q", output, limit=1) == {"found": 3, "written": 1}
        assert calls[-1]["rows"] == 1
    assert records[0] == {
        "bibcode": "A",
        "links_data": [{"access": "open", "type": "pdf", "url": "https://example.test/a.pdf"}],
    }
    assert [record["bibcode"] for record in records] == ["A", "B", "C"]
    assert records[1]["links_data"] == []


def test_jsonl_tolerates_truncated_last_line() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "results.jsonl"
        path.write_text('{"id":1}\n{"id":2}\n{"id":', encoding="utf-8")
        assert read_jsonl(path) == [{"id": 1}, {"id": 2}]
        append_jsonl(path, {"id": 3})
        assert path.read_text(encoding="utf-8") == '{"id":1}\n{"id":2}\n{"id":3}\n'

        path.write_text('{"id":1}', encoding="utf-8")
        append_jsonl(path, {"id": 2})
        assert read_jsonl(path) == [{"id": 1}, {"id": 2}]

        path.write_text('{"id":1}\n{"id":\n{"id":3}\n', encoding="utf-8")
        with pytest.raises(ValueError, match=re.escape(f"{path}:2:")):
            read_jsonl(path)

        path.write_text('{"id":1}\n{"id":\n', encoding="utf-8")  # a complete last line: edited by hand, not truncated
        with pytest.raises(ValueError, match=re.escape(f"{path}:2:")):
            read_jsonl(path)
