import json
from pathlib import Path

from chidi.graph import run_tutor_graph
from chidi.models import LearnerInput


DATASET_PATH = Path("data/eval/golden.jsonl")


def main() -> None:
    passed = 0
    failed = 0

    with DATASET_PATH.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            case = json.loads(line)
            learner = LearnerInput.model_validate(case["learner"])
            reply = run_tutor_graph(learner)

            expected_action = case["expected_action"]

            if reply.action == expected_action:
                print(f"PASS {case['case_id']}: action is {reply.action}")
                passed += 1
            else:
                print(
                    f"FAIL {case['case_id']}: "
                    f"expected {expected_action}, got {reply.action}"
                )
                failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
