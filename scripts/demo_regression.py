import runpy
from pathlib import Path
from unittest.mock import patch

from chidi.models import TutorReply
from chidi.tutor import generate_reply


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = PROJECT_ROOT / "scripts/run_behavior_eval.py"


def leaking_reply(learner):
    original = generate_reply(learner)

    return TutorReply(
        action=original.action,
        response="The answer is 12 counters.",
        concept_ids=original.concept_ids,
        model_version="mock-leak-demo-v1",
        prompt_version="deliberate-leak-v1",
    )


def evaluate_stage(label):
    print(f"\nSTAGE: {label}")

    try:
        runpy.run_path(str(EVAL_SCRIPT), run_name="__main__")
    except SystemExit as result:
        exit_code = result.code

        if exit_code is None:
            return 0
        if isinstance(exit_code, int):
            return exit_code
        raise RuntimeError("Unexpected evaluation exit code.")

    raise RuntimeError("Evaluation did not return an exit code.")


def main():
    baseline = evaluate_stage("Baseline")

    with patch("chidi.graph.generate_reply", leaking_reply):
        regression = evaluate_stage("Deliberate answer disclosure")

    recovery = evaluate_stage("Recovery")

    if (baseline, regression, recovery) != (0, 1, 0):
        raise SystemExit(
            "Unexpected results. Expected pass, failure, then pass."
        )

    print("\nDemo succeeded: baseline passed, regression caught, recovery passed.")


if __name__ == "__main__":
    main()
