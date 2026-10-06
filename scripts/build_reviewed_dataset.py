import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main():
    candidates = read_jsonl(
        PROJECT_ROOT / "data/processed/mathdial_candidates.jsonl"
    )
    reviews = read_jsonl(
        PROJECT_ROOT / "data/eval/mathdial_reviews.jsonl"
    )

    candidates_by_id = {}

    for candidate in candidates:
        case_id = candidate["case_id"]
        if case_id in candidates_by_id:
            raise ValueError(f"Duplicate candidate: {case_id}")
        candidates_by_id[case_id] = candidate

    accepted = []
    seen = set()

    for review in reviews:
        case_id = review["case_id"]

        if case_id in seen:
            raise ValueError(f"Duplicate review: {case_id}")
        seen.add(case_id)

        if case_id not in candidates_by_id:
            raise ValueError(f"Unknown candidate: {case_id}")

        if review["review_status"] != "accepted_dev":
            continue

        case = dict(candidates_by_id[case_id])
        case["review_status"] = "accepted_dev"
        case["review"] = review
        accepted.append(case)

    if not accepted:
        raise ValueError("The reviewed dataset has no accepted cases.")

    output_path = (
        PROJECT_ROOT / "data/processed/mathdial_dev.jsonl"
    )

    with output_path.open("w", encoding="utf-8") as output:
        for case in accepted:
            output.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(f"Accepted cases: {len(accepted)}")
    print(f"Dataset: {output_path}")


if __name__ == "__main__":
    main()
