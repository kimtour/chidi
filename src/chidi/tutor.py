"""A simple mock tutor for practising the Chidi workflow."""

from chidi.models import LearnerInput, TutorReply


def generate_reply(learner: LearnerInput) -> TutorReply:
    """Choose a question based on whether the learner has attempted the problem."""

    if learner.attempt is None:
        return TutorReply(
            action="ask",
            response=(
                "What have you tried so far? "
                "Tell me what the numbers represent and what you need to find."
            ),
            model_version="mock-v1",
        )

    return TutorReply(
        action="guide",
        response=(
            "What does each number in your attempt represent? "
            "Which part are you trying to work out first?"
        ),
        model_version="mock-v1",
    )
