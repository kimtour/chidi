from uuid import uuid4

from chidi.graph import run_tutor_graph
from chidi.models import LearnerInput


def main() -> None:
    session_id = f"session-{uuid4().hex}"

    print("Chidi learning companion")
    print("Type quit at the help question to finish.")

    problem = input("Enter your learning problem: ").strip()

    if not problem:
        print("Please enter a problem and run the program again.")
        return

    for turn in range(1, 11):
        print(f"\nTurn {turn}")

        learner_message = input(
            "What do you need help with? "
        ).strip()

        if learner_message.lower() == "quit":
            print("Session finished.")
            return

        if not learner_message:
            learner_message = "Please help me understand this problem."

        attempt = input(
            "Your current attempt, or Enter if none: "
        ).strip() or None

        learner = LearnerInput(
            session_id=session_id,
            problem=problem,
            learner_message=learner_message,
            attempt=attempt,
        )

        reply = run_tutor_graph(learner)
        print(f"\nTutor ({reply.action}): {reply.response}")

    print("\nSession reached its limit of 10 turns.")


if __name__ == "__main__":
    main()
