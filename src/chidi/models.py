"""Validated inputs and outputs for the Chidi tutor."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LearnerInput(BaseModel):
    """Information supplied by the learner."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    problem: str = Field(min_length=1)
    learner_message: str = Field(min_length=1)
    attempt: str | None = None

    @field_validator("session_id", "problem", "learner_message")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("This field must contain text.")

        return cleaned

    @field_validator("attempt")
    @classmethod
    def clean_attempt(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()

        return cleaned or None


class TutorReply(BaseModel):
    """Information returned by the tutor."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["ask", "guide", "hint", "escalate"]
    response: str = Field(min_length=1)

    concept_ids: list[str] = Field(default_factory=list)

    model_version: str = "mock-v1"
    prompt_version: str = "socratic-v1"

    @field_validator("response")
    @classmethod
    def reject_blank_response(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("The tutor response must contain text.")

        return cleaned
