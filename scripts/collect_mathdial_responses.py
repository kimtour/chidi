import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from chidi.graph import run_tutor_graph
from chidi.models import LearnerInput


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(
        description="Collect tutor responses for reviewed MathDial cases."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use paid model requests. Default is mock mode.",
    )
    args = parser.parse_args()

    dataset_path = PROJECT_ROOT / "data/processed/mathdial_dev.jsonl"
    dataset_bytes = dataset_path.read_bytes()
    dataset_hash = hashlib.sha256(dataset_bytes).hexdigest()

    cases = [
        json.loads(line)
        for line in dataset_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]

    if not cases:
        raise ValueError("The dataset is empty.")

    case_ids = [case["case_id"] for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("The dataset contains duplicate case IDs.")

    for case in cases:
        if case.get("review_status") != "accepted_dev":
            raise ValueError(f"Unreviewed case: {case['case_id']}")
        LearnerInput.model_validate(case["learner"])

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"mathdial-{timestamp}-{uuid4().hex[:8]}"
    report_path = PROJECT_ROOT / "reports" / f"{run_id}.jsonl"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    errors = 0

    with report_path.open("w", encoding="utf-8") as output:
        for case in cases:
            learner_data = dict(case["learner"])
            learner_data["session_id"] = f"{run_id}-{case['case_id']}"
            learner = LearnerInput.model_validate(learner_data)

            record = {
                "run_id": run_id,
                "case_id": case["case_id"],
                "mode": "live" if args.live else "mock",
                "dataset_sha256": dataset_hash,
                "source": case["source"],
                "learner": learner.model_dump(),
                "review": case["review"],
                "evaluator": case["evaluator"],
                "human_review_status": "pending",
            }

            try:
                reply = run_tutor_graph(learner, live=args.live)

                record.update({
                    "status": "generated",
                    "reply": reply.model_dump(),
                    "action_passed": (
                        reply.action == case["expected_action"]
                    ),
                })

                print(f"\nCASE: {case['case_id']}")
                print(f"Tutor: {reply.response}")

            except Exception as error:
                errors += 1
                record.update({
                    "status": "error",
                    "error_type": type(error).__name__,
                })
                print(
                    f"\nERROR {case['case_id']}: "
                    f"{type(error).__name__}"
                )

            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()

    print(f"\nCases: {len(cases)}")
    print(f"Generated: {len(cases) - errors}")
    print(f"Errors: {errors}")
    print(f"Report: {report_path}")
    print("Pedagogical quality still requires review.")

    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
