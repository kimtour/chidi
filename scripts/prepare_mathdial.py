import hashlib
import json
from pathlib import Path

from chidi.models import LearnerInput


INPUT_PATH = Path("data/raw/mathdial_train.jsonl")
OUTPUT_PATH = Path("data/processed/mathdial_candidates.jsonl")
SAMPLE_SIZE = 20


def main() -> None:
    source_hash = hashlib.sha256(INPUT_PATH.read_bytes()).hexdigest()
    cases = []
    seen_problem_ids = set()

    with INPUT_PATH.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)
            problem_id = str(record["qid"])

            if problem_id in seen_problem_ids:
                continue

            problem = record["question"].strip()
            attempt = record["student_incorrect_solution"].strip()

            if not problem or not attempt:
                continue

            learner = LearnerInput(
                session_id=f"mathdial-{problem_id}",
                problem=problem,
                learner_message="Please help me understand my mistake.",
                attempt=attempt,
            )

            cases.append({
                "case_id": f"mathdial-{problem_id}",
                "learner": learner.model_dump(),
                "expected_action": "guide",
                "review_status": "needs_review",
                "evaluator": {
                    "reference_solution": record["ground_truth"],
                    "described_confusion": record.get(
                        "teacher_described_confusion"
                    ),
                },
                "source": {
                    "dataset": "eth-nlped/mathdial",
                    "split": "train",
                    "problem_id": problem_id,
                    "file_sha256": source_hash,
                },
            })

            seen_problem_ids.add(problem_id)

            if len(cases) == SAMPLE_SIZE:
                break

    if not cases:
        raise ValueError("The dataset produced zero usable cases.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for case in cases:
            file.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(f"Prepared {len(cases)} candidate cases.")
    print(f"Saved to {OUTPUT_PATH}")
    print("Each case needs review before becoming a golden example.")


if __name__ == "__main__":
    main()
