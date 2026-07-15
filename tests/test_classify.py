from types import SimpleNamespace

from iris_paper_llm.classify import IRIS_PROMPT_SHA256, IRIS_SYSTEM_PROMPT, classify_paper
from iris_paper_llm.models import (
    Decision,
    Evidence,
    IRISAspect,
    IRISClassification,
    PaperChunk,
    ResultStatus,
    SyntheticConnection,
)


def classification(chunk_id: str = "chunk-7") -> IRISClassification:
    return IRISClassification(
        observational_use=Decision.NO,
        synthetic_use=Decision.YES,
        synthetic_connection=SyntheticConnection.PASSBAND_ONLY,
        review_only=Decision.NO,
        iris_mission_mentioned=False,
        aspects=[IRISAspect.SPECTROGRAPH],
        observational_evidence=[],
        synthetic_evidence=[Evidence(page=None, chunk_id=chunk_id, reason="Synthetic Mg II spectrum.")],
        review_evidence=[],
    )


def response(parsed: IRISClassification | None, *, status: str = "completed") -> SimpleNamespace:
    return SimpleNamespace(
        id="response-test",
        model="test-model",
        output=[],
        output_parsed=parsed,
        status=status,
        incomplete_details="max_output_tokens" if status != "completed" else None,
        usage=None,
    )


def test_prompt_and_api_contract() -> None:
    class FakeResponses:
        def parse(self, **kwargs: object) -> object:
            self.kwargs = kwargs
            return response(classification())

    responses = FakeResponses()
    result = classify_paper(
        [PaperChunk(page=None, chunk_id="chunk-7", text="We synthesize the Mg II k line.")],
        pdf_sha256="0" * 64,
        retrieval_mode="auto",
        client=SimpleNamespace(responses=responses),
    )
    assert result.provenance.prompt_sha256 == IRIS_PROMPT_SHA256
    assert responses.kwargs["text_format"] is IRISClassification
    assert "page=unknown chunk_id=chunk-7" in responses.kwargs["input"][1]["content"]
    assert "1331.7-1358.4 Angstrom" in IRIS_SYSTEM_PROMPT
    assert "```" not in IRIS_SYSTEM_PROMPT


def test_incomplete_and_invented_evidence_fail() -> None:
    chunk = PaperChunk(page=None, chunk_id="chunk-7", text="Some text")
    incomplete = classify_paper(
        [chunk],
        pdf_sha256="1" * 64,
        retrieval_mode="auto",
        client=SimpleNamespace(responses=SimpleNamespace(parse=lambda **_kwargs: response(None, status="incomplete"))),
    )
    assert incomplete.status == ResultStatus.PROCESSING_FAILED
    assert "incomplete" in incomplete.errors[0]

    rejected = classify_paper(
        [chunk],
        pdf_sha256="2" * 64,
        retrieval_mode="auto",
        client=SimpleNamespace(
            responses=SimpleNamespace(parse=lambda **_kwargs: response(classification("invented-chunk")))
        ),
    )
    assert rejected.status == ResultStatus.PROCESSING_FAILED
    assert "unknown chunk ID" in rejected.errors[0]


if __name__ == "__main__":
    test_prompt_and_api_contract()
    test_incomplete_and_invented_evidence_fail()
