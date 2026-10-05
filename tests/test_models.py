import pytest
from pydantic import ValidationError

from chidi.models import LearnerInput, TutorReply


def test_valid_learner_input():
    learner = LearnerInput(
        session_id="session-001",
        problem="There are 3 groups of 4 counters.",
        learner_message="Help me choose the operation.",
        attempt="I tried adding 3 and 4.",
    )

    assert learner.session_id == "session-001"
    assert learner.attempt == "I tried adding 3 and 4."


def test_blank_problem_is_rejected():
    with pytest.raises(ValidationError):
        LearnerInput(
            session_id="session-001",
            problem="   ",
            learner_message="Help me.",
        )


def test_empty_attempt_becomes_none():
    learner = LearnerInput(
        session_id="session-001",
        problem="There are 3 groups of 4 counters.",
        learner_message="Where should I start?",
        attempt="   ",
    )

    assert learner.attempt is None


def test_invalid_tutor_action_is_rejected():
    with pytest.raises(ValidationError):
        TutorReply(
            action="give_every_answer",
            response="Here is the solution.",
        )


def test_valid_tutor_reply_is_cleaned():
    reply = TutorReply(
        action="ask",
        response="  What have you tried so far?  ",
    )

    assert reply.action == "ask"
    assert reply.response == "What have you tried so far?"
    assert reply.concept_ids == []


def test_reference_solution_is_rejected():
    with pytest.raises(ValidationError):
        LearnerInput(
            session_id="session-001",
            problem="There are 3 groups of 4 counters.",
            learner_message="Help me.",
            reference_solution="12",
        )
