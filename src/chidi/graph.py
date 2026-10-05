from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from chidi.models import LearnerInput, TutorReply
from chidi.tutor import generate_reply


class ChidiState(TypedDict):
    learner: LearnerInput
    reply: NotRequired[TutorReply]


def make_reply(state: ChidiState) -> dict[str, TutorReply]:
    reply = generate_reply(state["learner"])
    return {"reply": reply}


builder = StateGraph(ChidiState)

builder.add_node("make_reply", make_reply)
builder.add_edge(START, "make_reply")
builder.add_edge("make_reply", END)

chidi_graph = builder.compile()


def run_tutor_graph(learner: LearnerInput) -> TutorReply:
    result = chidi_graph.invoke({"learner": learner})
    return result["reply"]
