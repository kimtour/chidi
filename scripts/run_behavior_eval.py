import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from chidi.graph import run_tutor_graph
from chidi.models import LearnerInput
from chidi.scorers import check_answer_disclosure


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check action selection and possible answer disclosure."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use live model requests instead of the mock tutor.",
    )
    args = parser.parse_args()

    dataset_path = PROJECT_ROOT / "data/eval/behavior.jsonl"
    cases = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not cases:
        raise ValueError("The evaluation dataset is empty.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"behavior-{timestamp}-{uuid4().hex[:8]}"
    records = []

    for case in cases:
        record = {
            "run_id": run_id,
            "case_id": case["case_id"],
            "mode": "live" if args.live else "mock",
            "scorer_version": "answer-match-v1",
        }

        try:
            learner = LearnerInput(
                session_id=f"{run_id}-{case['case_id']}",
                problem=case["problem"],
                learner_message=case["learner_message"],
                attempt=case["attempt"],
            )

            reply = run_tutor_graph(learner, live=args.live)

            disclosure = check_answer_disclosure(
                reply.response,
                case["answer_aliases"],
                allow_answer=case["allow_answer"],
            )

            action_passed = reply.action == case["expected_action"]
            status = (
                "pass"
                if action_passed and not disclosure["flagged"]
                else "fail"
            )

            record.update({
                "status": status,
                "learner": learner.model_dump(),
                "reply": reply.model_dump(),
                "expected_action": case["expected_action"],
                "action_passed": action_passed,
                "disclosure": disclosure,
            })

            print(
                f"{status.upper()} {case['case_id']}: "
                f"action={reply.action}, "
                f"disclosure_flag={disclosure['flagged']}"
            )

        except Exception as error:
            record.update({
                "status": "error",
                "error_type": type(error).__name__,
            })
            print(f"ERROR {case['case_id']}: {type(error).__name__}")

        records.append(record)

    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{run_id}.jsonl"

    with report_path.open("w", encoding="utf-8") as report:
        for record in records:
            report.write(json.dumps(record, ensure_ascii=False) + "\n")

    passed = sum(record["status"] == "pass" for record in records)
    failed = sum(record["status"] == "fail" for record in records)
    errors = sum(record["status"] == "error" for record in records)

    print(f"\nCases: {len(records)}")
    print(f"Results: {passed} passed, {failed} failed, {errors} errors")
    print(f"Report: {report_path}")

    raise SystemExit(1 if failed or errors else 0)


if __name__ == "__main__":
    main()
