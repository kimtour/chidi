import argparse

from chidi.retrieval import ConceptStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    args = parser.parse_args()

    store = ConceptStore()
    try:
        results = store.retrieve(args.query)

        if not results:
            print("No concepts returned.")

        for result in results:
            print(
                f"\n{result['concept_id']} "
                f"(similarity={result['score']:.3f})"
            )
            print(result["text"])
    finally:
        store.close()


if __name__ == "__main__":
    main()
