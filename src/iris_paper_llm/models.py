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


class Basis(StrEnum):
    OBSERVATIONAL_DATA = "OBSERVATIONAL_DATA"
    SYNTHETIC_OBSERVABLE = "SYNTHETIC_OBSERVABLE"
    INSTRUMENT_OR_SOFTWARE = "INSTRUMENT_OR_SOFTWARE"
    COMPANION_PAPER = "COMPANION_PAPER"
    REVIEW = "REVIEW"


class ResultStatus(StrEnum):
    CLASSIFIED = "CLASSIFIED"
    PROCESSING_FAILED = "PROCESSING_FAILED"


class Evidence(StrictModel):
    page: int = Field(ge=1)
    reason: str = Field(min_length=1)


class IRISClassification(StrictModel):
    include: Decision
    basis: list[Basis]
    iris_mission_mentioned: bool
    evidence: list[Evidence]

    @model_validator(mode="after")
    def validate_consistency(self) -> IRISClassification:
        if len(self.basis) != len(set(self.basis)):
            msg = "basis must not contain duplicates"
            raise ValueError(msg)
        if self.include == Decision.NO and self.basis:
            msg = "include=NO requires an empty basis"
            raise ValueError(msg)
        if self.include == Decision.YES and not set(self.basis) - {Basis.REVIEW}:
            msg = "include=YES requires at least one basis other than REVIEW"
            raise ValueError(msg)
        if self.include == Decision.UNCERTAIN and set(self.basis) - {Basis.REVIEW}:
            msg = "include=UNCERTAIN allows only the REVIEW basis"
            raise ValueError(msg)
        if self.basis and not self.evidence:
            msg = f"include={self.include} with basis {self.basis} requires evidence"
            raise ValueError(msg)
        return self


class PipelineProvenance(StrictModel):
    pipeline_version: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: str | None
    reasoning_effort: str = Field(min_length=1)
    request_id: str | None
    input_tokens: int | None = Field(ge=0)
    output_tokens: int | None = Field(ge=0)


class PaperResult(StrictModel):
    schema_version: Literal[2] = 2
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
