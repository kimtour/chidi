import asyncio
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from openai import AsyncOpenAI
from dotenv import dotenv_values
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

from chidi.graph import chidi_graph, run_tutor_graph
from chidi.models import LearnerInput


ROOT = Path(__file__).resolve().parents[1]


async def main():
    settings = dotenv_values(ROOT / ".env")
    key = (settings.get("ANTHROPIC_API_KEY") or "").strip()
    model = (settings.get("RAGAS_MODEL") or "").strip()

    if not key or not model:
        raise ValueError("API key and judge model are required.")

    learner = LearnerInput(
        session_id=f"ragas-demo-{uuid4().hex}",
        problem=(
            "Wanda walks half a mile to school and half a mile home "
            "in both the morning and afternoon. How far does she "
            "walk in five school days?"
        ),
        learner_message="Help me understand my mistake.",
        attempt="One round trip is 1 mile, so five days is 5 miles.",
    )

    reply = run_tutor_graph(learner, live=True)

    snapshot = chidi_graph.get_state({
        "configurable": {"thread_id": learner.session_id}
    })
    concepts = snapshot.values.get("concepts", [])
    retrieved_contexts = [concept["text"] for concept in concepts]

    if not retrieved_contexts:
        raise ValueError("Retrieved concept evidence is empty.")

    combined_contexts = retrieved_contexts + [
        f"Learner problem: {learner.problem}",
        f"Learner attempt: {learner.attempt}",
    ]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = ROOT / "reports" / (
        f"ragas-{timestamp}-{uuid4().hex[:8]}.jsonl"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print("Concepts supplied:", reply.concept_ids)
    print("Tutor:", reply.response)

    errors = 0

    async with AsyncOpenAI(
        api_key=key,
        base_url="https://openrouter.ai/api/v1",
        timeout=60.0,
        max_retries=0,
    ) as client:
        llm = llm_factory(
            model,
            provider="openai",
            client=client,
            max_tokens=2048,
        )
        metric = Faithfulness(llm=llm)

        scopes = {
            "retrieved_concepts_only": retrieved_contexts,
            "concepts_plus_learner_information": combined_contexts,
        }

        with report_path.open("w", encoding="utf-8") as output:
            for scope, contexts in scopes.items():
                record = {
                    "session_id": learner.session_id,
                    "metric": "faithfulness",
                    "ragas_version": "0.4.3",
                    "evidence_scope": scope,
                    "judge_model_requested": model,
                    "learner": learner.model_dump(),
                    "reply": reply.model_dump(),
                    "retrieved_concepts": concepts,
                    "evaluation_contexts": contexts,
                }

                try:
                    result = await metric.ascore(
                        user_input=learner.problem,
                        response=reply.response,
                        retrieved_contexts=contexts,
                    )

                    score = float(result.value)

                    if not math.isfinite(score):
                        record.update({
                            "status": "not_applicable",
                            "score": None,
                            "reason": "Metric returned a non-finite score.",
                        })
                        print(f"\n{scope}: not applicable")
                    elif not 0 <= score <= 1:
                        raise ValueError("Faithfulness score is out of range.")
                    else:
                        record.update({
                            "status": "scored",
                            "score": score,
                        })
                        print(f"\n{scope}: {score:.3f}")

                except Exception as error:
                    errors += 1
                    record.update({
                        "status": "error",
                        "score": None,
                        "error_type": type(error).__name__,
                    })
                    print(f"\n{scope}: ERROR {type(error).__name__}")

                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()

    print(f"\nErrors: {errors}")
    print(f"Report: {report_path}")
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    asyncio.run(main())
