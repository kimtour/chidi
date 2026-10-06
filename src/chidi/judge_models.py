from pydantic import BaseModel, ConfigDict, Field, StrictBool


class CriterionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: StrictBool
    rationale: str = Field(min_length=1)


class JudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accuracy: CriterionResult
    misconception_handling: CriterionResult
    single_focused_question: CriterionResult
    premature_disclosure: CriterionResult
    needs_human_review: StrictBool
    review_reason: str
