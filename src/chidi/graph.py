from typing import Annotated, Literal, NotRequired, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from chidi.models import LearnerInput, TutorReply
from chidi.tutor import generate_reply


class ChidiState(TypedDict):
    learner: dict
    messages: Annotated[list[AnyMessage], add_messages]
    reply: NotRequired[dict]
    live: NotRequired[bool]
    concepts: NotRequired[list[dict]]


def retrieve_concepts(state: ChidiState) -> dict:
    if not state.get("live", False):
        return {"concepts": []}

    learner = LearnerInput.model_validate(state["learner"])

    from chidi.scope import is_course_information_request

    if is_course_information_request(learner.problem):
        return {"concepts": []}

    from chidi.retrieval import ConceptStore

    query = (
        f"Problem: {learner.problem}\n"
        f"Attempt: {learner.attempt or ''}\n"
        f"Help request: {learner.learner_message}"
    )

    store = ConceptStore()
    try:
        concepts = store.retrieve(query, limit=1)
    finally:
        store.close()

    return {"concepts": concepts}


def choose_route(state: ChidiState) -> Literal["ask", "guide"]:
    learner = LearnerInput.model_validate(state["learner"])
    return "ask" if learner.attempt is None else "guide"


def make_reply(state: ChidiState) -> dict:
    learner = LearnerInput.model_validate(state["learner"])

    if state.get("live", False):
        from chidi.live_tutor import generate_live_reply

        reply = generate_live_reply(
            learner,
            state["messages"],
            concepts=state.get("concepts", []),
        )
    else:
        reply = generate_reply(learner)

    return {
        "reply": reply.model_dump(),
        "messages": [AIMessage(content=reply.response)],
    }


def build_graph():
    builder = StateGraph(ChidiState)

    builder.add_node("retrieve", retrieve_concepts)
    builder.add_node("ask", make_reply)
    builder.add_node("guide", make_reply)

    builder.add_edge(START, "retrieve")
    builder.add_conditional_edges(
        "retrieve",
        choose_route,
        {"ask": "ask", "guide": "guide"},
    )
    builder.add_edge("ask", END)
    builder.add_edge("guide", END)

    return builder.compile(checkpointer=InMemorySaver())


chidi_graph = build_graph()


def run_tutor_graph(
    learner: LearnerInput, *, live: bool = False
) -> TutorReply:
    result = chidi_graph.invoke(
        {
            "learner": learner.model_dump(),
            "messages": [
                HumanMessage(content=learner.model_dump_json())
            ],
            "live": live,
        },
        {"configurable": {"thread_id": learner.session_id}},
    )

    return TutorReply.model_validate(result["reply"])
