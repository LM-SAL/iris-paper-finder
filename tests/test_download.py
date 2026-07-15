from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import fitz

from iris_paper_llm.download import candidate_pdf_urls, download_records, read_jsonl, validate_pdf


def one_page_pdf() -> bytes:
    with fitz.open() as document:
        document.new_page()
        return document.tobytes()


class FakeResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.status_code = 200
        self.headers = {"content-type": "application/pdf"}
        self.url = "https://example.test/paper.pdf"


class FakeSession:
    def __init__(self, content: bytes) -> None:
        self.response = FakeResponse(content)
        self.calls = 0

    def get(self, *_args, **_kwargs) -> FakeResponse:
        self.calls += 1
        return self.response


def test_download_is_atomic_validated_and_resumable() -> None:
    content = one_page_pdf()
    session = FakeSession(content)
    record = {"bibcode": "test-paper", "pdf_links": ["https://example.test/paper.pdf"]}
    with TemporaryDirectory() as directory:
        output = Path(directory)
        first = download_records([record], output, session=session)
        validate_pdf((output / "test-paper.pdf").read_bytes())
        assert first == {"downloaded": 1, "skipped": 0, "failed": 0, "manual_queue": 0}
        assert read_jsonl(output / "download_attempts.jsonl")[0]["status"] == "downloaded"

        second = download_records([record], output, session=session)
        assert second == {"downloaded": 0, "skipped": 1, "failed": 0, "manual_queue": 0}
        assert session.calls == 1


def test_ads_link_fallback_returns_a_list() -> None:
    record = {
        "bibcode": "test-paper",
        "links_data": [
            {
                "access": "open",
                "type": "preprint",
                "url": "http://arxiv.org/abs/2501.01234",
            }
        ],
    }
    assert candidate_pdf_urls(record) == ["https://arxiv.org/pdf/2501.01234"]


def test_manual_queue_is_deduplicated() -> None:
    session = FakeSession(one_page_pdf())
    with TemporaryDirectory() as directory:
        output = Path(directory)
        record = {"bibcode": "missing-paper", "links_data": []}
        expected = {"downloaded": 0, "skipped": 0, "failed": 1, "manual_queue": 1}
        assert download_records([record], output, browser_fallback=True, session=session) == expected
        assert download_records([record], output, browser_fallback=True, session=session) == expected
        queue = read_jsonl(output / "manual_downloads.jsonl")
        assert len(queue) == 1
        assert queue[0]["category"] == "permanent"
        assert session.calls == 0


if __name__ == "__main__":
    test_download_is_atomic_validated_and_resumable()
    test_ads_link_fallback_returns_a_list()
    test_manual_queue_is_deduplicated()
