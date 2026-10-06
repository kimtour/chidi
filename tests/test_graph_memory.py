from langchain_core.messages import HumanMessage

from chidi.graph import build_graph
from chidi.models import LearnerInput


def invoke_turn(graph, session_id, attempt=None):
    learner = LearnerInput(
        session_id=session_id,
        problem="There are 3 groups of 4 counters.",
        learner_message="Help me understand.",
        attempt=attempt,
    )

    return graph.invoke(
        {
            "learner": learner.model_dump(),
            "messages": [
                HumanMessage(content=learner.model_dump_json())
            ],
        },
        {"configurable": {"thread_id": session_id}},
    )


def test_two_turns_preserve_history():
    graph = build_graph()

    first = invoke_turn(graph, "same-session")
    second = invoke_turn(graph, "same-session", "I added 3 and 4.")

    assert first["reply"]["action"] == "ask"
    assert second["reply"]["action"] == "guide"
    assert len(second["messages"]) == 4
    assert [message.type for message in second["messages"]] == [
        "human", "ai", "human", "ai"
    ]


def test_sessions_have_separate_history():
    graph = build_graph()

    invoke_turn(graph, "session-a", "I added 3 and 4.")
    result = invoke_turn(graph, "session-b")

    assert len(result["messages"]) == 2
    assert result["learner"]["session_id"] == "session-b"
    assert result["reply"]["action"] == "ask"
