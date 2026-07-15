from pydantic import ValidationError

from iris_paper_llm.models import Decision, Evidence, IRISAspect, IRISClassification, SyntheticConnection


def classification() -> IRISClassification:
    return IRISClassification(
        observational_use=Decision.NO,
        synthetic_use=Decision.YES,
        synthetic_connection=SyntheticConnection.PASSBAND_ONLY,
        review_only=Decision.NO,
        iris_mission_mentioned=True,
        aspects=[IRISAspect.SPECTROGRAPH],
        observational_evidence=[],
        synthetic_evidence=[Evidence(page=2, chunk_id="p2-c1", reason="Synthetic Mg II intensity is analyzed.")],
        review_evidence=[],
    )


def test_classification_consistency() -> None:
    record = classification().model_dump(mode="json")
    invalid_records = [
        {**record, "synthetic_connection": "NOT_APPLICABLE"},
        {
            **record,
            "review_only": "YES",
            "review_evidence": [{"page": 2, "chunk_id": "p2-c1", "reason": "Prior work."}],
        },
    ]
    for invalid in invalid_records:
        try:
            IRISClassification.model_validate(invalid)
        except ValidationError:
            pass
        else:
            raise AssertionError("inconsistent classification was accepted")


if __name__ == "__main__":
    test_classification_consistency()
