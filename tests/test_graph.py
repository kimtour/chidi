from chidi.graph import run_tutor_graph
from chidi.models import LearnerInput


def test_graph_returns_a_question_when_learner_has_no_attempt():
    learner = LearnerInput(
        session_id="session-003",
        problem="There are 3 groups of 4 counters.",
        learner_message="Help me.",
    )

    reply = run_tutor_graph(learner)

    assert reply.action == "ask"
    assert "tried" in reply.response.lower()


def test_graph_returns_guidance_when_learner_has_an_attempt():
    learner = LearnerInput(
        session_id="session-004",
        problem="There are 3 groups of 4 counters.",
        learner_message="Is my answer right?",
        attempt="I added 3 and 4.",
    )

    reply = run_tutor_graph(learner)

    assert reply.action == "guide"
    assert reply.model_version == "mock-v1"
