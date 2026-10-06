from typing import Annotated, Literal, NotRequired, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from chidi.models import LearnerInput, TutorReply
from chidi.tutor import generate_reply


class ChidiState(TypedDict):
    learner: LearnerInput
    messages: Annotated[list[AnyMessage], add_messages]
    reply: NotRequired[TutorReply]


def choose_route(state: ChidiState) -> Literal["ask", "guide"]:
    if state["learner"].attempt is None:
        return "ask"
    return "guide"


def make_reply(state: ChidiState) -> dict:
    reply = generate_reply(state["learner"])

    return {
        "reply": reply,
        "messages": [AIMessage(content=reply.response)],
    }


def build_graph():
    builder = StateGraph(ChidiState)

    builder.add_node("ask", make_reply)
    builder.add_node("guide", make_reply)

    builder.add_conditional_edges(
        START,
        choose_route,
        {"ask": "ask", "guide": "guide"},
    )

    builder.add_edge("ask", END)
    builder.add_edge("guide", END)

    return builder.compile(checkpointer=InMemorySaver())


chidi_graph = build_graph()


def run_tutor_graph(learner: LearnerInput) -> TutorReply:
    result = chidi_graph.invoke(
        {
            "learner": learner,
            "messages": [
                HumanMessage(content=learner.model_dump_json())
            ],
        },
        {"configurable": {"thread_id": learner.session_id}},
    )

    return result["reply"]
