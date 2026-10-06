import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from chidi.retrieval import ConceptStore, MODEL_NAME


ROOT = Path(__file__).resolve().parents[1]


def main():
    dataset_path = ROOT / "data/eval/retrieval.jsonl"
    cases = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not cases:
        raise ValueError("Retrieval evaluation dataset is empty.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = ROOT / "reports" / (
        f"retrieval-{timestamp}-{uuid4().hex[:8]}.jsonl"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)

    passed = 0
    failed = 0
    errors = 0
    store = ConceptStore()

    try:
        with report_path.open("w", encoding="utf-8") as output:
            for case in cases:
                record = {
                    **case,
                    "embedding_model": MODEL_NAME,
                    "corpus_version": "math-concepts-v1",
                    "limit": 1,
                }

                try:
                    results = store.retrieve(case["query"], limit=1)
                    retrieved_ids = [
                        result["concept_id"] for result in results
                    ]
                    hit = case["expected_concept"] in retrieved_ids

                    passed += int(hit)
                    failed += int(not hit)

                    record.update({
                        "status": "pass" if hit else "fail",
                        "retrieved": results,
                    })

                    print(
                        f"{record['status'].upper()} "
                        f"{case['case_id']}: {retrieved_ids}"
                    )

                except Exception as error:
                    errors += 1
                    record.update({
                        "status": "error",
                        "error_type": type(error).__name__,
                    })
                    print(
                        f"ERROR {case['case_id']}: "
                        f"{type(error).__name__}"
                    )

                output.write(json.dumps(record) + "\n")
                output.flush()
    finally:
        store.close()

    print(f"\nCases: {len(cases)}")
    print(f"Results: {passed} passed, {failed} failed, {errors} errors")
    print(f"Top-1 hits across all cases: {passed}/{len(cases)}")
    print(f"Report: {report_path}")

    raise SystemExit(1 if failed or errors else 0)


if __name__ == "__main__":
    main()
