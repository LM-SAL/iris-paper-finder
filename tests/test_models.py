import pytest
from pydantic import ValidationError

from iris_paper_llm.models import IRISClassification

EVIDENCE = [{"page": 2, "reason": "SYNTHETIC_OBSERVABLE: synthetic Mg II intensity is analyzed."}]


def test_classification_consistency() -> None:
    valid = [
        {"include": "YES", "basis": ["SYNTHETIC_OBSERVABLE", "REVIEW"], "evidence": EVIDENCE},
        {"include": "NO", "basis": [], "evidence": []},
        {"include": "UNCERTAIN", "basis": ["REVIEW"], "evidence": EVIDENCE},
        {"include": "UNCERTAIN", "basis": [], "evidence": []},
    ]
    invalid = [
        {"include": "YES", "basis": [], "evidence": EVIDENCE},
        {"include": "YES", "basis": ["REVIEW"], "evidence": EVIDENCE},
        {"include": "YES", "basis": ["OBSERVATIONAL_DATA"], "evidence": []},
        {"include": "YES", "basis": ["OBSERVATIONAL_DATA", "OBSERVATIONAL_DATA"], "evidence": EVIDENCE},
        {"include": "NO", "basis": ["REVIEW"], "evidence": EVIDENCE},
        {"include": "UNCERTAIN", "basis": ["OBSERVATIONAL_DATA"], "evidence": EVIDENCE},
        {"include": "UNCERTAIN", "basis": ["REVIEW"], "evidence": []},
        {"include": "YES", "basis": ["OBSERVATIONAL_DATA"], "evidence": [{"page": 0, "reason": "p0"}]},
    ]
    for record in valid:
        IRISClassification.model_validate({**record, "iris_mission_mentioned": True})
    for record in invalid:
        with pytest.raises(ValidationError):
            IRISClassification.model_validate({**record, "iris_mission_mentioned": True})
