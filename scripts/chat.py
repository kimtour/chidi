from uuid import uuid4

from chidi.graph import run_tutor_graph
from chidi.models import LearnerInput


def main() -> None:
    session_id = f"session-{uuid4().hex[:8]}"

    print("Chidi learning companion")
    problem = input("Enter the problem you are working on: ").strip()

    if not problem:
        print("Please enter a problem and run the program again.")
        return

    learner_message = input("What do you need help with? ").strip()

    if not learner_message:
        learner_message = "Please help me understand this problem."

    attempt_text = input(
        "What have you tried so far? Press Enter if you have not tried yet: "
    ).strip()

    attempt = attempt_text or None

    learner = LearnerInput(
        session_id=session_id,
        problem=problem,
        learner_message=learner_message,
        attempt=attempt,
    )

    reply = run_tutor_graph(learner)

    print(f"\nTutor ({reply.action}): {reply.response}")


if __name__ == "__main__":
    main()
