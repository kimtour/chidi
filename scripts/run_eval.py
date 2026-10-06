import argparse
import json
from pathlib import Path

from chidi.graph import run_tutor_graph
from chidi.models import LearnerInput


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Chidi tutor actions.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/eval/golden.jsonl"),
        help="Path to a JSONL evaluation file.",
    )
    args = parser.parse_args()

    passed = 0
    failed = 0
    errors = 0
    total = 0

    with args.dataset.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            total += 1

            try:
                case = json.loads(line)
                case_id = case["case_id"]
                expected_action = case["expected_action"]

                if expected_action not in {"ask", "guide", "hint", "escalate"}:
                    raise ValueError("Invalid expected_action.")

                learner = LearnerInput.model_validate(case["learner"])
                reply = run_tutor_graph(learner)

                if reply.action == expected_action:
                    passed += 1
                    print(f"PASS {case_id}: action is {reply.action}")
                else:
                    failed += 1
                    print(
                        f"FAIL {case_id}: "
                        f"expected {expected_action}, got {reply.action}"
                    )

            except Exception as error:
                errors += 1
                print(
                    f"ERROR line {line_number}: "
                    f"{type(error).__name__}: {error}"
                )

    print(f"\nDataset: {args.dataset}")
    print(f"Cases: {total}")
    print(f"Results: {passed} passed, {failed} failed, {errors} errors")

    if total == 0:
        print("Evaluation failed: the dataset contains zero cases.")
        raise SystemExit(1)

    if failed or errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
