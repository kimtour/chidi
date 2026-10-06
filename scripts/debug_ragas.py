import asyncio
import json
import traceback
from pathlib import Path

from openai import AsyncOpenAI
from dotenv import dotenv_values
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness


ROOT = Path(__file__).resolve().parents[1]


async def main():
    path = ROOT / "reports/ragas-20261006T090915Z-d79970d3.jsonl"
    record = json.loads(
        path.read_text(encoding="utf-8").splitlines()[0]
    )

    settings = dotenv_values(ROOT / ".env")
    key = (settings.get("ANTHROPIC_API_KEY") or "").strip()
    model = (settings.get("RAGAS_MODEL") or "").strip()

    if not key or not model:
        raise ValueError("API key and judge model are required.")

    try:
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

            result = await Faithfulness(llm=llm).ascore(
                user_input=record["learner"]["problem"],
                response=record["reply"]["response"],
                retrieved_contexts=record["evaluation_contexts"],
            )

            print("Result type:", type(result).__name__)
            print("Result:", result)

    except Exception:
        diagnostic = traceback.format_exc()

        for name, value in settings.items():
            if value and any(
                marker in name.upper()
                for marker in ("KEY", "TOKEN", "SECRET", "PASSWORD")
            ):
                diagnostic = diagnostic.replace(value, "[REDACTED]")

        print(diagnostic)


if __name__ == "__main__":
    asyncio.run(main())
