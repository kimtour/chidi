from chidi.retrieval import ConceptStore


def main():
    store = ConceptStore()
    try:
        count = store.ingest()
        print(f"Indexed {count} concept notes.")
    finally:
        store.close()


if __name__ == "__main__":
    main()
