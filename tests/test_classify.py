from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import httpx2
import openai
import pymupdf
import pytest

from iris_paper_llm.classify import (
    IRIS_PROMPT_SHA256,
    classify_paper,
    extract_pages,
    remove_reference_section,
)
from iris_paper_llm.models import Basis, Decision, Evidence, IRISClassification, ResultStatus

PAGES = ["We synthesize the Mg II k line.", "", "The Mg II k intensity is compared."]


def classification(page: int = 3) -> IRISClassification:
    return IRISClassification(
        include=Decision.YES,
        basis=[Basis.SYNTHETIC_OBSERVABLE],
        iris_mission_mentioned=False,
        evidence=[Evidence(page=page, reason="SYNTHETIC_OBSERVABLE: synthetic Mg II spectrum.")],
    )


def response(parsed: IRISClassification | None, *, status: str = "completed") -> SimpleNamespace:
    return SimpleNamespace(
        id="response-test",
        model="test-model",
        output=[SimpleNamespace(type="reasoning", content=None)],
        output_parsed=parsed,
        status=status,
        incomplete_details="max_output_tokens" if status != "completed" else None,
        usage=None,
    )


class FakeResponses:
    def __init__(self, *results: object) -> None:
        self.results = list(results)
        self.calls = []

    def parse(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def classify(*results: object) -> tuple:
    responses = FakeResponses(*results)
    result = classify_paper(PAGES, pdf_hash="0" * 64, client=SimpleNamespace(responses=responses))
    return result, responses.calls


def test_prompt_and_api_contract() -> None:
    result, calls = classify(response(classification()))
    assert result.status == ResultStatus.CLASSIFIED
    assert result.provenance.prompt_sha256 == IRIS_PROMPT_SHA256
    assert result.provenance.reasoning_effort == "medium"
    assert calls[0]["text_format"] is IRISClassification
    assert calls[0]["reasoning"] == {"effort": "medium"}
    context = calls[0]["input"][1]["content"]
    assert context.startswith("[PAGE 1]\nWe synthesize")
    assert "[PAGE 2]" not in context
    assert "[PAGE 3]\nThe Mg II k intensity is compared.\n[END PAGE 3]" in context


def test_single_retry_on_bad_evidence_page() -> None:
    recovered, calls = classify(response(classification(page=2)), response(classification(page=1)))
    assert recovered.status == ResultStatus.CLASSIFIED
    assert len(calls) == 2
    assert "[2]" in recovered.errors[0]

    rejected, calls = classify(response(classification(page=2)), response(classification(page=9)))
    assert rejected.status == ResultStatus.PROCESSING_FAILED
    assert len(calls) == 2
    assert "[2]" in rejected.errors[0]
    assert "[9]" in rejected.errors[1]

    incomplete, calls = classify(response(None, status="incomplete"))
    assert incomplete.status == ResultStatus.PROCESSING_FAILED
    assert "incomplete" in incomplete.errors[0]
    assert len(calls) == 1


def test_account_wide_errors_are_raised() -> None:
    request = httpx2.Request("POST", "https://api.openai.test/v1/responses")
    authentication = openai.AuthenticationError(
        "invalid key", response=httpx2.Response(401, request=request), body=None
    )
    quota = openai.RateLimitError(
        "quota", response=httpx2.Response(429, request=request), body={"code": "insufficient_quota"}
    )
    effort = openai.BadRequestError(
        "unsupported value", response=httpx2.Response(400, request=request), body={"param": "reasoning.effort"}
    )
    for error in (authentication, quota, effort):
        with pytest.raises(type(error)):
            classify(error)

    throttled = openai.RateLimitError("slow down", response=httpx2.Response(429, request=request), body=None)
    too_long = openai.BadRequestError(
        "too long", response=httpx2.Response(400, request=request), body={"param": "input"}
    )
    for error in (throttled, too_long):
        result, _calls = classify(error)
        assert result.status == ResultStatus.PROCESSING_FAILED


def test_reference_removal_keeps_appendix() -> None:
    pages = [
        "Introduction\nsee Smith et al. and\nreferences therein.\nReferences therein use IRIS data.",
        "Results\n5. References\nAlpha, A. 2020, ApJ",
        "Beta, B. 2021, A&A\nAppendix A: Line fitting\nWe fit Mg II k.",
        "Appendix B\nMore.",
    ]
    assert remove_reference_section(pages) == [
        pages[0],
        "Results",
        "Appendix A: Line fitting\nWe fit Mg II k.",
        pages[3],
    ]
    assert remove_reference_section([*pages[:2], "Beta, B. 2021, A&A"]) == [pages[0], "Results", ""]
    assert remove_reference_section(pages[:1]) == pages[:1]
    nature = ["Results\nReferences\nAlpha, A. 2020, ApJ", "Beta, B. 2021\nMethods\nIRIS observed the flare."]
    assert remove_reference_section(nature) == ["Results", "Methods\nIRIS observed the flare."]
    preprint = ["Results\nREFERENCES\nAlpha, A. 2020, ApJ", "Beta, B. 2021\nFig. 2.— IRIS SJI 1400 image."]
    assert remove_reference_section(preprint) == ["Results", "Fig. 2.— IRIS SJI 1400 image."]


def test_whitespace_collapse() -> None:
    with pymupdf.open() as document, TemporaryDirectory() as directory:
        page = document.new_page()
        page.insert_text((72, 72), "We   analyze\tIRIS    data.   ")
        for y in (100, 120, 140):
            page.insert_text((72, y), "   ")
        page.insert_text((72, 200), "The Mg II k line is fitted in every raster position.")
        page.insert_text((72, 220), "Its intensity is compared with synthetic profiles.")
        path = Path(directory) / "paper.pdf"
        document.save(path)
        assert extract_pages(path) == [
            (
                "We analyze IRIS data.\n\nThe Mg II k line is fitted in every raster position.\n"
                "Its intensity is compared with synthetic profiles."
            )
        ]
