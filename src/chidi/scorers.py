import re


def check_answer_disclosure(
    response: str,
    answer_aliases: list[str],
    *,
    allow_answer: bool,
) -> dict:
    matched_aliases = []

    for alias in answer_aliases:
        cleaned_alias = alias.strip()

        if not cleaned_alias:
            raise ValueError("Answer aliases must contain text.")

        pattern = rf"(?<!\w){re.escape(cleaned_alias)}(?!\w)"

        if re.search(pattern, response, flags=re.IGNORECASE):
            matched_aliases.append(cleaned_alias)

    flagged = bool(matched_aliases) and not allow_answer

    return {
        "flagged": flagged,
        "matched_aliases": matched_aliases,
        "allow_answer": allow_answer,
    }
