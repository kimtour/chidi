import pytest

from chidi.graph import retrieve_concepts
from chidi.scope import is_course_information_request


@pytest.mark.parametrize(
    "problem",
    [
        "What does my next course module cover?",
        "Where is the course outline?",
        "Help me find the module topics.",
    ],
)
def test_detects_course_information_requests(problem):
    assert is_course_information_request(problem) is True


def test_preserves_normal_math_scope():
    assert is_course_information_request(
        "Wanda walks to school in the morning and afternoon."
    ) is False


def test_course_request_bypasses_retrieval():
    result = retrieve_concepts({
        "live": True,
        "learner": {
            "session_id": "scope-regression",
            "problem": "What does my next course module cover?",
            "learner_message": "Help me find the module topics.",
            "attempt": None,
        },
        "messages": [],
    })

    assert result == {"concepts": []}
