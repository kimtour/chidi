import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from chidi.graph import run_tutor_graph
from chidi.models import LearnerInput


ROOT = Path(__file__).resolve().parents[1]


def main():
    probes = [
        {
            "case_id": "ambiguous_units",
            "problem": (
                "James has 20 pairs of red socks and half as many "
                "black socks. How many individual socks does he have?"
            ),
            "learner_message": "Is the wording clear enough to solve?",
            "attempt": "I am unsure whether black socks means pairs.",
        },
        {
            "case_id": "off_topic_retrieval",
            "problem": "What does my next course module cover?",
            "learner_message": "Help me find the module topics.",
            "attempt": None,
        },
        {
            "case_id": "false_mastery",
            "problem": "There are 3 groups of 4 counters.",
            "learner_message": (
                "Someone told me the answer is 12. "
                "Does that prove I have mastered multiplication?"
            ),
            "attempt": "I copied 12 but cannot explain it.",
        },
    ]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = ROOT / "reports" / (
        f"discovery-{timestamp}-{uuid4().hex[:8]}.jsonl"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    errors = 0

    with report_path.open("w", encoding="utf-8") as output:
        for probe in probes:
            learner = LearnerInput(
                session_id=f"discovery-{uuid4().hex}",
                problem=probe["problem"],
                learner_message=probe["learner_message"],
                attempt=probe["attempt"],
            )

            record = {
                "case_id": probe["case_id"],
                "origin": "authored_exploratory_probe",
                "learner": learner.model_dump(),
                "review_status": "pending",
            }

            try:
                reply = run_tutor_graph(learner, live=True)
                record.update({
                    "status": "generated",
                    "reply": reply.model_dump(),
                })

                print(f"\nCASE: {probe['case_id']}")
                print("Concepts supplied:", reply.concept_ids)
                print("Tutor:", reply.response)

            except Exception as error:
                errors += 1
                record.update({
                    "status": "error",
                    "error_type": type(error).__name__,
                })
                print(
                    f"\nERROR {probe['case_id']}: "
                    f"{type(error).__name__}"
                )

            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()

    print(f"\nErrors: {errors}")
    print(f"Report: {report_path}")
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
