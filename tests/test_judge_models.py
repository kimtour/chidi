from copy import deepcopy

import pytest
from pydantic import ValidationError

from chidi.judge_models import JudgeResult


def valid_payload():
    return {
        "accuracy": {
            "value": True,
            "rationale": "The mathematical guidance is correct.",
        },
        "misconception_handling": {
            "value": True,
            "rationale": "The response addresses the extra segment.",
        },
        "single_focused_question": {
            "value": False,
            "rationale": "The response asks two questions.",
        },
        "premature_disclosure": {
            "value": False,
            "rationale": "The final total is withheld.",
        },
        "needs_human_review": False,
        "review_reason": "",
    }


def test_accepts_complete_judgment():
    result = JudgeResult.model_validate(valid_payload())

    assert result.accuracy.value is True
    assert result.single_focused_question.value is False


@pytest.mark.parametrize("invalid_value", ["true", "false", 1, 0, None])
def test_rejects_non_boolean_decisions(invalid_value):
    payload = deepcopy(valid_payload())
    payload["accuracy"]["value"] = invalid_value

    with pytest.raises(ValidationError):
        JudgeResult.model_validate(payload)


def test_rejects_missing_criterion():
    payload = valid_payload()
    del payload["accuracy"]

    with pytest.raises(ValidationError):
        JudgeResult.model_validate(payload)


def test_rejects_unexpected_field():
    payload = valid_payload()
    payload["overall_score"] = 100

    with pytest.raises(ValidationError):
        JudgeResult.model_validate(payload)


def test_rejects_empty_rationale():
    payload = valid_payload()
    payload["accuracy"]["rationale"] = ""

    with pytest.raises(ValidationError):
        JudgeResult.model_validate(payload)
