from pathlib import Path
from uuid import uuid4

from dotenv import dotenv_values
from langfuse import Langfuse

from chidi.graph import run_tutor_graph
from chidi.models import LearnerInput
from chidi.scorers import check_answer_disclosure


ROOT = Path(__file__).resolve().parents[1]


def main():
    settings = dotenv_values(ROOT / ".env")

    required = [
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_BASE_URL",
    ]

    for name in required:
        if not (settings.get(name) or "").strip():
            raise ValueError(f"{name} is required.")

    langfuse = Langfuse(
        public_key=settings["LANGFUSE_PUBLIC_KEY"],
        secret_key=settings["LANGFUSE_SECRET_KEY"],
        base_url=settings["LANGFUSE_BASE_URL"],
    )

    learner = LearnerInput(
        session_id=f"langfuse-demo-{uuid4().hex}",
        problem="There are 3 groups of 4 counters. How many altogether?",
        learner_message="Help me understand my mistake.",
        attempt="I added 3 and 4.",
    )

    try:
        with langfuse.start_as_current_observation(
            as_type="span",
            name="chidi-tutor-turn",
            input=learner.model_dump(),
            metadata={
                "session_id": learner.session_id,
                "source": "learning-demo",
            },
        ) as observation:
            reply = run_tutor_graph(learner, live=True)

            disclosure = check_answer_disclosure(
                reply.response,
                ["12", "twelve"],
                allow_answer=False,
            )

            observation.update(
                output=reply.model_dump(),
                metadata={
                    "session_id": learner.session_id,
                    "model_version": reply.model_version,
                    "prompt_version": reply.prompt_version,
                    "concept_ids": reply.concept_ids,
                    "scorer_version": "answer-match-v1",
                },
            )

            langfuse.create_score(
                trace_id=observation.trace_id,
                name="answer_disclosure_flag",
                value=int(disclosure["flagged"]),
                data_type="BOOLEAN",
                comment=(
                    "Exact answer matching. A flagged response "
                    "requires contextual review."
                ),
            )

            print("Tutor:", reply.response)
            print("Disclosure flag:", disclosure["flagged"])
            print("Trace ID:", observation.trace_id)

    finally:
        langfuse.flush()

    print("Open the chidi project in Langfuse to inspect the trace.")


if __name__ == "__main__":
    main()
