import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    examples = [
        {
            "case_id": "focused_hint",
            "response": (
                "Adding the group count to the group size mixes two "
                "different quantities. What addition expression would "
                "represent three groups containing four counters each?"
            ),
            "labels": [True, True, True, False],
            "notes": "Accurate, targeted guidance with one question and no answer.",
        },
        {
            "case_id": "wrong_guidance",
            "response": (
                "Your addition method is correct because a group count "
                "and a group size should be added to find the total."
            ),
            "labels": [False, False, True, False],
            "notes": (
                "Incorrectly endorses the misconception. Gives one bounded "
                "instructional statement without revealing the correct answer."
            ),
        },
        {
            "case_id": "unrelated_feedback",
            "response": (
                "Your calculation of 3 + 4 = 7 is arithmetically correct. "
                "What is your favourite school subject?"
            ),
            "labels": [True, False, True, False],
            "notes": (
                "The arithmetic statement is accurate, but the response "
                "does not address why addition is inappropriate."
            ),
        },
        {
            "case_id": "full_solution",
            "response": (
                "You should multiply instead of adding the group count "
                "and group size. The total is 3 x 4 = 12 counters."
            ),
            "labels": [True, True, True, True],
            "notes": (
                "Accurate and targeted, but supplies the complete solution "
                "before the learner works through the misconception."
            ),
        },
    ]

    fields = [
        "accuracy",
        "misconception_handling",
        "single_focused_question",
        "premature_disclosure",
    ]

    report_path = ROOT / "reports/judge_calibration_responses.jsonl"
    labels_path = ROOT / "reports/judge_calibration_labels.jsonl"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if report_path.exists() or labels_path.exists():
        raise SystemExit(
            "Calibration files already exist. Existing files preserved."
        )

    with (
        report_path.open("w", encoding="utf-8") as reports,
        labels_path.open("w", encoding="utf-8") as labels,
    ):
        for example in examples:
            case_id = example["case_id"]
            response = example["response"]

            record = {
                "run_id": "synthetic-calibration-v1",
                "case_id": case_id,
                "status": "generated",
                "origin": "synthetic_authored",
                "learner": {
                    "session_id": case_id,
                    "problem": (
                        "There are 3 groups of 4 counters. "
                        "How many counters are there altogether?"
                    ),
                    "learner_message": "Help me understand my mistake.",
                    "attempt": "I added 3 and 4.",
                },
                "reply": {
                    "response": response,
                    "model_version": "authored-fixture",
                },
                "review": {
                    "reference_answer": "12",
                    "misconception": (
                        "Adds group count and group size "
                        "instead of combining equal groups."
                    ),
                    "expected_behavior": (
                        "Guide the learner toward equal groups "
                        "without supplying the complete solution."
                    ),
                },
                "evaluator": {
                    "reference_solution": "4 + 4 + 4 = 12 counters.",
                },
            }

            label = {
                "run_id": record["run_id"],
                "case_id": case_id,
                "response_sha256": hashlib.sha256(
                    response.encode("utf-8")
                ).hexdigest(),
                "rubric_version": "pedagogy-v1",
                "reviewer": "AI-authored synthetic reference labels",
                "notes": example["notes"],
                **dict(zip(fields, example["labels"])),
            }

            reports.write(json.dumps(record) + "\n")
            labels.write(json.dumps(label) + "\n")

    print(f"Responses: {report_path}")
    print(f"Labels: {labels_path}")
    print("Created four synthetic calibration examples.")


if __name__ == "__main__":
    main()
