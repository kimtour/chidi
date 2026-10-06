import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from anthropic import Anthropic
from dotenv import dotenv_values

from chidi.judge_models import JudgeResult


ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    "accuracy",
    "misconception_handling",
    "single_focused_question",
    "premature_disclosure",
]


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main():
    report_path = ROOT / (
        "reports/mathdial-20261006T082851Z-66595716.jsonl"
    )
    labels_path = ROOT / (
        "reports/mathdial-20261006T082851Z-66595716-human-review.jsonl"
    )

    records = read_jsonl(report_path)
    labels = read_jsonl(labels_path)
    labels_by_id = {}

    for label in labels:
        case_id = label["case_id"]
        if case_id in labels_by_id:
            raise ValueError(f"Duplicate label: {case_id}")
        labels_by_id[case_id] = label

    if not records:
        raise ValueError("Response report is empty.")

    seen = set()

    for record in records:
        case_id = record["case_id"]

        if case_id in seen:
            raise ValueError(f"Duplicate response: {case_id}")
        seen.add(case_id)

        if record["status"] != "generated":
            raise ValueError(f"Missing response: {case_id}")

        label = labels_by_id[case_id]
        response_hash = hashlib.sha256(
            record["reply"]["response"].encode("utf-8")
        ).hexdigest()

        if (
            label["run_id"] != record["run_id"]
            or label["response_sha256"] != response_hash
            or label["rubric_version"] != "pedagogy-v1"
        ):
            raise ValueError(f"Review does not match response: {case_id}")

        for field in FIELDS:
            if type(label.get(field)) is not bool:
                raise ValueError(f"Incomplete label: {case_id}/{field}")

    if set(labels_by_id) != seen:
        raise ValueError("Response and label case IDs differ.")

    settings = dotenv_values(ROOT / ".env")
    key = (settings.get("ANTHROPIC_API_KEY") or "").strip()
    model = (settings.get("CHIDI_JUDGE_MODEL") or "").strip()

    if not key or not model:
        raise ValueError("API key and judge model are required.")

    client = Anthropic(
        api_key="",
        auth_token=key,
        base_url=(
            settings.get("ANTHROPIC_BASE_URL")
            or "https://openrouter.ai/api"
        ),
        timeout=60.0,
        max_retries=0,
    )

    rubric = (ROOT / "prompts/judge.txt").read_text(encoding="utf-8")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = ROOT / "reports" / (
        f"judge-{timestamp}-{uuid4().hex[:8]}.jsonl"
    )

    agreements = {field: 0 for field in FIELDS}
    completed = 0
    errors = 0
    review_flags = 0

    with output_path.open("w", encoding="utf-8") as output:
        for record in records:
            case_id = record["case_id"]
            label = labels_by_id[case_id]

            payload = {
                "learner": record["learner"],
                "tutor_response": record["reply"]["response"],
                "reviewed_reference": record["review"],
                "source_reference": record["evaluator"],
            }

            saved = {
                "case_id": case_id,
                "source_run_id": record["run_id"],
                "response_sha256": label["response_sha256"],
                "judge_model_requested": model,
                "rubric_version": "pedagogy-v1",
                "rubric_sha256": hashlib.sha256(
                    rubric.encode("utf-8")
                ).hexdigest(),
                "comparison_reviewer": label["reviewer"],
            }

            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=2048,
                    system=rubric,
                    messages=[{
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False),
                    }],
                )

                if response.stop_reason != "end_turn":
                    raise ValueError("Judge response did not finish normally.")

                text = "\n".join(
                    block.text
                    for block in response.content
                    if block.type == "text"
                )

                verdict = JudgeResult.model_validate_json(text)

                matches = {
                    field: getattr(verdict, field).value == label[field]
                    for field in FIELDS
                }

                for field, matched in matches.items():
                    agreements[field] += int(matched)

                completed += 1
                review_flags += int(verdict.needs_human_review)

                saved.update({
                    "status": "judged",
                    "judge_model_returned": response.model,
                    "verdict": verdict.model_dump(),
                    "comparison_labels": {
                        field: label[field] for field in FIELDS
                    },
                    "agreement": matches,
                    "usage": response.usage.model_dump(mode="json"),
                })

                print(f"\nCASE: {case_id}")
                for field in FIELDS:
                    criterion = getattr(verdict, field)
                    print(
                        f"{field}: judge={criterion.value}, "
                        f"review={label[field]}, "
                        f"agree={matches[field]}"
                    )
                    print(f"  Reason: {criterion.rationale}")

            except Exception as error:
                errors += 1
                saved.update({
                    "status": "error",
                    "error_type": type(error).__name__,
                })
                print(f"\nERROR {case_id}: {type(error).__name__}")

            output.write(json.dumps(saved, ensure_ascii=False) + "\n")
            output.flush()

    print(f"\nJudged: {completed}/{len(records)}")
    print(f"Errors: {errors}")
    print(f"Human-review flags: {review_flags}")

    for field in FIELDS:
        print(f"{field} agreement: {agreements[field]}/{completed}")

    print(f"Report: {output_path}")
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
