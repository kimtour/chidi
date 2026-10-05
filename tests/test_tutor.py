from chidi.models import LearnerInput
from chidi.tutor import generate_reply


def test_tutor_asks_for_an_attempt_when_none_is_given():
    learner = LearnerInput(
        session_id="session-001",
        problem="There are 3 groups of 4 counters.",
        learner_message="I don't know where to begin.",
    )

    reply = generate_reply(learner)

    assert reply.action == "ask"
    assert "tried" in reply.response.lower()
    assert reply.model_version == "mock-v1"


def test_tutor_guides_the_learner_when_an_attempt_is_given():
    learner = LearnerInput(
        session_id="session-002",
        problem="There are 3 groups of 4 counters.",
        learner_message="Is my answer right?",
        attempt="I added 3 and 4.",
    )

    reply = generate_reply(learner)

    assert reply.action == "guide"
    assert "number" in reply.response.lower()
    assert reply.model_version == "mock-v1"
