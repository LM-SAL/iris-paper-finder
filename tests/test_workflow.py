from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import fitz

from iris_paper_llm.ads import _metadata_for_bibcodes, iris_query
from iris_paper_llm.classify import DEFAULT_MODEL, classify_paths
from iris_paper_llm.evaluate import prepare_case_pdfs, write_report
from iris_paper_llm.models import (
    Decision,
    Evidence,
    IRISAspect,
    IRISClassification,
    PaperResult,
    ResultStatus,
    RetrievalMode,
    SyntheticConnection,
)


def paper_pdf(label: str) -> bytes:
    with fitz.open() as document:
        page = document.new_page()
        page.insert_textbox(
            fitz.Rect(72, 72, 500, 700),
            f"{label}. We analyze IRIS observations. " * 20,
        )
        return document.tobytes()


class FakeResponse:
    status_code = 200
    url = "https://example.test/paper.pdf"

    def __init__(self, content: bytes) -> None:
        self.content = content
        self.headers = {"content-type": "application/pdf"}


class FakeSession:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def get(self, *_args, **_kwargs) -> FakeResponse:
        return FakeResponse(self.content)


class FakeEmbedder:
    def create_embeddings(self, docs: list) -> None:
        self.docs = docs

    def get_relevant_docs(self, _query: str, kwargs: dict | None = None, n_results: int = 10):
        docs = self.docs
        if kwargs and kwargs.get("where"):
            docs = [doc for doc in docs if doc.metadata["is_candidate"] == 1]
        docs = docs[:n_results]
        return docs, [float(index) for index in range(len(docs))]


class CharacterEncoding:
    @staticmethod
    def encode(text: str, **_kwargs: object) -> list[int]:
        return [ord(character) for character in text]

    @staticmethod
    def decode(tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


class FakeResponses:
    def parse(self, **kwargs: object) -> object:
        context = kwargs["input"][1]["content"]
        page = int(context.split("page=", 1)[1].split(" ", 1)[0])
        chunk_id = context.split("chunk_id=", 1)[1].split("]", 1)[0]
        classification = IRISClassification(
            observational_use=Decision.YES,
            synthetic_use=Decision.NO,
            synthetic_connection=SyntheticConnection.NOT_APPLICABLE,
            review_only=Decision.NO,
            iris_mission_mentioned=True,
            aspects=[IRISAspect.TELESCOPE],
            observational_evidence=[Evidence(page=page, chunk_id=chunk_id, reason="IRIS data are analyzed.")],
            synthetic_evidence=[],
            review_evidence=[],
        )
        return SimpleNamespace(
            id="response-test",
            model="resolved-test-model",
            output=[],
            output_parsed=classification,
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
    second = repository / "tests/data/pdfs/instrument_description_only.pdf"
    with TemporaryDirectory() as directory:
        root = Path(directory)
        pdfs = root / "pdfs"
        pdfs.mkdir()
        shutil.copyfile(first, pdfs / first.name)
        shutil.copyfile(second, pdfs / second.name)
        output = root / "results.jsonl"

        with patch("tiktoken.get_encoding", return_value=CharacterEncoding()):
            first_summary = classify_paths(first, output, client=client, embedder=FakeEmbedder())
            assert len(output.read_text().splitlines()) == 1
            directory_summary = classify_paths(pdfs, output, client=client, embedder=FakeEmbedder())

            failed_output = root / "failed.jsonl"
            failed_summary = classify_paths(
                first,
                failed_output,
                client=SimpleNamespace(responses=FailingResponses()),
                embedder=FakeEmbedder(),
            )

        assert first_summary == {"positive": 1, "negative": 0, "uncertain": 0, "failed": 0, "skipped": 0}
        assert directory_summary == {"positive": 1, "negative": 0, "uncertain": 0, "failed": 0, "skipped": 1}
        records = [json.loads(line) for line in output.read_text().splitlines()]
        assert len(records) == 2
        sent = records[0]["retrieval"]["sent"]
        result = PaperResult.model_validate(records[0]["result"])
        assert sent
        assert result.status == ResultStatus.CLASSIFIED
        assert result.classification is not None
        assert result.classification.overall == Decision.YES
        evidence = result.classification.observational_evidence[0]
        assert (evidence.chunk_id, evidence.page) == (sent[0]["chunk_id"], sent[0]["page"])

        assert failed_summary == {"positive": 0, "negative": 0, "uncertain": 0, "failed": 1, "skipped": 0}
        failed = PaperResult.model_validate(json.loads(failed_output.read_text())["result"])
        assert failed.status == ResultStatus.PROCESSING_FAILED
        assert failed.classification is None
        report = write_report(
            output,
            model=DEFAULT_MODEL,
            retrieval_mode=RetrievalMode.AUTO,
            top_k=20,
        )
        assert report["positive"] == 2
        assert output.with_suffix(".md").is_file()


def test_reviewed_pdf_preparation_uses_manifest_checksum() -> None:
    content = paper_pdf("Reviewed")
    checksum = hashlib.sha256(content).hexdigest()
    with TemporaryDirectory() as directory:
        root = Path(directory)
        cases = root / "cases.jsonl"
        case = {
            "id": "reviewed",
            "path": "pdfs/TEST.pdf",
            "pdf_sha256": checksum,
            "pdf_links": ["https://example.test/paper.pdf"],
            "bibcode": "TEST",
        }
        cases.write_text(json.dumps(case) + "\n")
        summary = prepare_case_pdfs(cases, base_dir=root, session=FakeSession(content))
        assert summary == {"downloaded": 1, "skipped": 0, "failed": 0, "manual_queue": 0}
        assert hashlib.sha256((root / "pdfs/TEST.pdf").read_bytes()).hexdigest() == checksum


def test_standard_query_is_scoped_to_year() -> None:
    query = iris_query(2025)
    assert "2014SoPh..289.2733D" in query
    assert "pubdate:[2025-01 TO 2025-12]" in query


def test_ads_metadata_normalizes_link_records() -> None:
    response = SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {
            "response": {
                "start": 0,
                "numFound": 1,
                "docs": [
                    {
                        "bibcode": "TEST",
                        "links_data": ['{"access":"open","type":"pdf","url":"https://example.test/paper.pdf"}'],
                    }
                ],
            }
        },
    )
    with patch("iris_paper_llm.ads.requests.post", return_value=response):
        records = _metadata_for_bibcodes(["TEST"], "token")
    assert records == [
        {
            "bibcode": "TEST",
            "links_data": [{"access": "open", "type": "pdf", "url": "https://example.test/paper.pdf"}],
            "pdf_links": [],
        }
    ]


if __name__ == "__main__":
    test_offline_pipeline_checkpoint_and_resume()
    test_reviewed_pdf_preparation_uses_manifest_checksum()
    test_standard_query_is_scoped_to_year()
    test_ads_metadata_normalizes_link_records()
