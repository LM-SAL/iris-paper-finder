"""Validated records for IRIS paper classification."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    exact_match_chunk_ids: list[str]
    adjacent_chunk_ids: list[str]
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
