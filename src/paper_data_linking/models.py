"""Validated records for IRIS paper classification."""

# ruff: noqa: S101

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Decision(StrEnum):
    YES = "YES"
    NO = "NO"
    UNCERTAIN = "UNCERTAIN"


class SyntheticConnection(StrEnum):
    EXPLICIT = "EXPLICIT"
    PASSBAND_ONLY = "PASSBAND_ONLY"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNCERTAIN = "UNCERTAIN"


class IRISAspect(StrEnum):
    TELESCOPE = "TELESCOPE"
    SPECTROGRAPH = "SPECTROGRAPH"
    SLIT_JAW_IMAGER = "SLIT_JAW_IMAGER"


class ResultStatus(StrEnum):
    CLASSIFIED = "CLASSIFIED"
    PROCESSING_FAILED = "PROCESSING_FAILED"
    NOT_ANALYZED = "NOT_ANALYZED"


class RetrievalMode(StrEnum):
    AUTO = "auto"
    HEURISTIC = "heuristic"
    ALL = "all"


class SelectionReason(StrEnum):
    HEURISTIC_MATCH = "heuristic_match"
    ADJACENT_CONTEXT = "adjacent_context"
    GLOBAL_FALLBACK = "global_fallback"


class Evidence(StrictModel):
    page: int | None = Field(ge=1)
    chunk_id: str = Field(min_length=1, pattern=r"^[^\r\n]+$")
    reason: str = Field(min_length=1)


class PaperChunk(StrictModel):
    page: int | None = Field(ge=1)
    chunk_id: str = Field(min_length=1, pattern=r"^[^\r\n]+$")
    text: str = Field(min_length=1)


class RetrievedChunk(PaperChunk):
    distance: float
    reason: SelectionReason


class RetrievalResult(StrictModel):
    mode: RetrievalMode
    query: str = Field(min_length=1)
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    top_k: int = Field(gt=0)
    exact_match_chunk_ids: list[str]
    adjacent_chunk_ids: list[str]
    excluded_chunk_ids: list[str]
    sent_chunk_ids: list[str]
    selected: list[RetrievedChunk]


class IRISClassification(StrictModel):
    observational_use: Decision
    synthetic_use: Decision
    synthetic_connection: SyntheticConnection
    review_only: Decision
    iris_mission_mentioned: bool
    aspects: list[IRISAspect]
    observational_evidence: list[Evidence]
    synthetic_evidence: list[Evidence]
    review_evidence: list[Evidence]

    @property
    def overall(self) -> Decision:
        if Decision.YES in {self.observational_use, self.synthetic_use}:
            return Decision.YES
        if self.observational_use == self.synthetic_use == Decision.NO:
            return Decision.NO
        return Decision.UNCERTAIN

    @model_validator(mode="after")
    def validate_consistency(self) -> IRISClassification:
        allowed_connections = {
            Decision.YES: {SyntheticConnection.EXPLICIT, SyntheticConnection.PASSBAND_ONLY},
            Decision.NO: {SyntheticConnection.NOT_APPLICABLE},
            Decision.UNCERTAIN: {SyntheticConnection.UNCERTAIN},
        }
        if self.synthetic_connection not in allowed_connections[self.synthetic_use]:
            msg = f"{self.synthetic_use=} is inconsistent with {self.synthetic_connection=}"
            raise ValueError(msg)

        if Decision.YES in {self.observational_use, self.synthetic_use} and self.review_only != Decision.NO:
            msg = "review_only must be NO when this paper has new observational or synthetic use"
            raise ValueError(msg)

        evidence_fields = (
            ("observational_use", self.observational_use, self.observational_evidence),
            ("synthetic_use", self.synthetic_use, self.synthetic_evidence),
            ("review_only", self.review_only, self.review_evidence),
        )
        for name, decision, evidence in evidence_fields:
            if decision in {Decision.YES, Decision.UNCERTAIN} and not evidence:
                msg = f"{name}={decision} requires evidence"
                raise ValueError(msg)

        if Decision.YES not in {self.observational_use, self.synthetic_use} and self.aspects:
            msg = "aspects must be empty when neither data-use field is YES"
            raise ValueError(msg)
        if len(self.aspects) != len(set(self.aspects)):
            msg = "aspects must not contain duplicates"
            raise ValueError(msg)
        return self


class PipelineProvenance(StrictModel):
    pipeline_version: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: str | None
    retrieval_mode: str | None
    request_id: str | None
    input_tokens: int | None = Field(ge=0)
    output_tokens: int | None = Field(ge=0)


class PaperResult(StrictModel):
    schema_version: Literal[1] = 1
    bibcode: str | None
    pdf_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: ResultStatus
    classification: IRISClassification | None
    provenance: PipelineProvenance
    errors: list[str]

    @model_validator(mode="after")
    def validate_status(self) -> PaperResult:
        if self.status == ResultStatus.CLASSIFIED and self.classification is None:
            msg = "CLASSIFIED requires a classification"
            raise ValueError(msg)
        if self.status != ResultStatus.CLASSIFIED and self.classification is not None:
            msg = f"{self.status} cannot contain a classification"
            raise ValueError(msg)
        if self.status == ResultStatus.PROCESSING_FAILED and not self.errors:
            msg = "PROCESSING_FAILED requires at least one error"
            raise ValueError(msg)
        if any(not error.strip() for error in self.errors):
            msg = "errors must not contain blank strings"
            raise ValueError(msg)
        return self


def _self_check() -> None:
    evidence = Evidence(page=2, chunk_id="page-2-chunk-1", reason="Synthetic Mg II intensity is analyzed.")
    classification = IRISClassification(
        observational_use=Decision.NO,
        synthetic_use=Decision.YES,
        synthetic_connection=SyntheticConnection.PASSBAND_ONLY,
        review_only=Decision.NO,
        iris_mission_mentioned=True,
        aspects=[IRISAspect.SPECTROGRAPH],
        observational_evidence=[],
        synthetic_evidence=[evidence],
        review_evidence=[],
    )
    provenance = PipelineProvenance(
        pipeline_version="phase1",
        prompt_version="iris-v3",
        prompt_sha256="0" * 64,
        model="test-model",
        retrieval_mode="auto",
        request_id="test-request",
        input_tokens=100,
        output_tokens=20,
    )
    result = PaperResult(
        bibcode="2025ApJ...978...27D",
        pdf_sha256="f293d40b872db3e58e9c4d791c38bcdbcf2fb037eb417faac1c57ae44327fdec",
        status=ResultStatus.CLASSIFIED,
        classification=classification,
        provenance=provenance,
        errors=[],
    )
    assert result.classification is not None
    assert result.classification.overall == Decision.YES
    assert PaperResult.model_validate_json(result.model_dump_json()) == result

    failed = PaperResult(
        bibcode=None,
        pdf_sha256="0" * 64,
        status=ResultStatus.PROCESSING_FAILED,
        classification=None,
        provenance=provenance,
        errors=["PDF extraction failed"],
    )
    assert failed.classification is None

    invalid = classification.model_dump(mode="json", exclude={"overall"})
    invalid["synthetic_connection"] = "NOT_APPLICABLE"
    try:
        IRISClassification.model_validate(invalid)
    except ValidationError:
        pass
    else:
        msg = "inconsistent synthetic connection was accepted"
        raise AssertionError(msg)

    invalid_review = classification.model_dump(mode="json")
    invalid_review["review_only"] = "YES"
    invalid_review["review_evidence"] = [evidence.model_dump(mode="json")]
    try:
        IRISClassification.model_validate(invalid_review)
    except ValidationError:
        pass
    else:
        msg = "review-only classification with new data use was accepted"
        raise AssertionError(msg)


if __name__ == "__main__":
    _self_check()
