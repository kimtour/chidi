import re


def is_course_information_request(problem: str) -> bool:
    patterns = [
        r"\b(?:course|next)\s+module\b",
        r"\bmodule\s+(?:topics|content|overview|covers?)\b",
        r"\bsyllabus\b",
        r"\bcourse\s+(?:outline|topics|content)\b",
    ]

    return any(
        re.search(pattern, problem, flags=re.IGNORECASE)
        for pattern in patterns
    )
