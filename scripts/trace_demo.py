from pathlib import Path
from uuid import uuid4

from dotenv import dotenv_values
from langsmith import Client, tracing_context

from chidi.graph import run_tutor_graph
from chidi.models import LearnerInput


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    settings = dotenv_values(root / ".env")

    key = (settings.get("LANGSMITH_API_KEY") or "").strip()
    if not key:
        raise ValueError("Configure LANGSMITH_API_KEY in .env.")

    client = Client(
        api_key=key,
        api_url=settings["LANGSMITH_ENDPOINT"],
        auto_batch_tracing=False,
    )

    session_id = f"trace-demo-{uuid4().hex}"
    attempts = [
        "I added 3 and 4.",
        "I drew 3 groups with 4 counters each and counted 12 counters.",
    ]

    with tracing_context(
        enabled=True,
        client=client,
        project_name=settings.get("LANGSMITH_PROJECT") or "chidi",
        tags=["learning-demo"],
        metadata={
            "session_id": session_id,
            "model_version": settings["ANTHROPIC_MODEL"],
            "prompt_version": "socratic-v2",
        },
    ):
        for turn, attempt in enumerate(attempts, start=1):
            learner = LearnerInput(
                session_id=session_id,
                problem="There are 3 groups of 4 counters.",
                learner_message="Please give feedback on my attempt.",
                attempt=attempt,
            )

            reply = run_tutor_graph(learner, live=True)
            print(f"\nTurn {turn}: {reply.response}")

    print(f"\nSession: {session_id}")
    print("Open the chidi tracing project in LangSmith.")


if __name__ == "__main__":
    main()
