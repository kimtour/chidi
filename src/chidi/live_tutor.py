from pathlib import Path

from anthropic import Anthropic
from langsmith import traceable
from dotenv import dotenv_values

from chidi.models import LearnerInput, TutorReply


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def generate_live_reply(learner: LearnerInput, history: list) -> TutorReply:
    settings = dotenv_values(PROJECT_ROOT / ".env")
    key = (settings.get("ANTHROPIC_API_KEY") or "").strip()
    model = (settings.get("ANTHROPIC_MODEL") or "").strip()

    if not key or not model:
        raise ValueError("Configure the API key and model in .env.")

    policy = (PROJECT_ROOT / "prompts/tutor.txt").read_text(
        encoding="utf-8"
    )

    messages = []
    for message in history:
        if message.type == "human":
            role = "user"
        elif message.type == "ai":
            role = "assistant"
        else:
            raise ValueError(f"Unsupported message type: {message.type}")

        messages.append({"role": role, "content": message.content})

    client = Anthropic(
        api_key="",
        auth_token=key,
        base_url=settings.get("ANTHROPIC_BASE_URL")
        or "https://openrouter.ai/api",
        timeout=60.0,
        max_retries=0,
    )

    traced_create = traceable(
        name="Chidi model request",
        run_type="llm",
    )(client.messages.create)

    result = traced_create(
        model=model,
        max_tokens=1024,
        system=policy,
        messages=messages,
    )

    if result.stop_reason != "end_turn":
        raise RuntimeError(
            f"Generation did not finish normally: {result.stop_reason}"
        )

    text = "\n".join(
        block.text for block in result.content if block.type == "text"
    ).strip()

    if not text:
        raise RuntimeError("The model returned no tutor text.")

    return TutorReply(
        action="ask" if learner.attempt is None else "guide",
        response=text,
        model_version=model,
        prompt_version="socratic-v3",
    )
