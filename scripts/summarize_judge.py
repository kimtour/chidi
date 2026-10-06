import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CRITERIA = [
    "accuracy",
    "misconception_handling",
    "single_focused_question",
    "premature_disclosure",
]


def main():
    parser = argparse.ArgumentParser(
        description="Summarize saved judge reports."
    )
    parser.add_argument("reports", nargs="+")
    args = parser.parse_args()

    summary_rows = []

    for filename in args.reports:
        path = ROOT / filename
        records = pd.read_json(path, lines=True)

        judged = records.loc[records["status"] == "judged"]
        errors = int((records["status"] == "error").sum())
        review_flags = sum(
            bool(verdict["needs_human_review"])
            for verdict in judged["verdict"]
        )

        print(f"\nREPORT: {path.name}")
        print(
            f"Cases: {len(records)}, judged: {len(judged)}, "
            f"errors: {errors}, review flags: {review_flags}"
        )

        for criterion in CRITERIA:
            values = [
                verdict[criterion]["value"]
                for verdict in judged["verdict"]
            ]
            matches = [
                agreement[criterion]
                for agreement in judged["agreement"]
            ]

            # Disclosure is undesirable; the other criteria are desirable.
            desired = criterion != "premature_disclosure"
            passing = sum(value == desired for value in values)
            agreement_count = sum(matches)
            count = len(values)

            summary_rows.append({
                "report": path.name,
                "criterion": criterion,
                "total_cases": len(records),
                "judged_cases": count,
                "errors": errors,
                "review_flags": review_flags,
                "criterion_passes": passing,
                "criterion_pass_rate": passing / count if count else None,
                "label_agreements": agreement_count,
                "label_agreement_rate": (
                    agreement_count / count if count else None
                ),
                "rubric_versions": ",".join(
                    sorted(set(judged["rubric_version"]))
                ),
            })

            print(
                f"{criterion}: "
                f"criterion passes {passing}/{count}, "
                f"label agreement {agreement_count}/{count}"
            )

    output_path = ROOT / "reports/judge_summary.csv"
    pd.DataFrame(summary_rows).to_csv(output_path, index=False)
    print(f"\nCSV saved: {output_path}")


if __name__ == "__main__":
    main()
