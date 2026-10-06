import argparse
from uuid import uuid4

from chidi.graph import run_tutor_graph
from chidi.models import LearnerInput


def ask_input(prompt: str) -> str:
    value = input(prompt).strip()

    if value.lower() == "quit":
        raise SystemExit("Session finished.")

    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with Chidi.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Generate replies through the configured API.",
    )
    args = parser.parse_args()

    session_id = f"session-{uuid4().hex}"

    print("Chidi learning companion")
    print(f"Mode: {'live' if args.live else 'mock'}")
    print("Type quit at any question to finish.")

    problem = ask_input("Enter your learning problem: ")

    if not problem:
        print("Please enter a problem and run the program again.")
        return

    for turn in range(1, 11):
        print(f"\nTurn {turn}")

        learner_message = ask_input("What do you need help with? ")
        attempt = ask_input(
            "Your current attempt, or Enter if none: "
        ) or None

        learner = LearnerInput(
            session_id=session_id,
            problem=problem,
            learner_message=learner_message
            or "Please help me understand this problem.",
            attempt=attempt,
        )

        reply = run_tutor_graph(learner, live=args.live)
        print(f"\nTutor ({reply.action}): {reply.response}")

    print("\nSession reached its limit of 10 turns.")


if __name__ == "__main__":
    main()
