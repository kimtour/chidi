from chidi.scorers import check_answer_disclosure


def test_flags_answer_before_learner_solves_problem():
    result = check_answer_disclosure(
        "The total is 12 counters.",
        ["12", "twelve"],
        allow_answer=False,
    )

    assert result["flagged"] is True
    assert result["matched_aliases"] == ["12"]


def test_allows_confirmation_of_correct_attempt():
    result = check_answer_disclosure(
        "Yes, 12 counters is correct.",
        ["12", "twelve"],
        allow_answer=True,
    )

    assert result["flagged"] is False
    assert result["matched_aliases"] == ["12"]


def test_detects_written_answer_case_insensitively():
    result = check_answer_disclosure(
        "There are Twelve counters.",
        ["12", "twelve"],
        allow_answer=False,
    )

    assert result["flagged"] is True


def test_avoids_matching_answer_inside_another_number():
    result = check_answer_disclosure(
        "Consider a different example with 120 counters.",
        ["12", "twelve"],
        allow_answer=False,
    )

    assert result["flagged"] is False
    assert result["matched_aliases"] == []
