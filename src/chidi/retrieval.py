import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models


ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION = "math_concepts_v1"


class ConceptStore:
    def __init__(self):
        self.client = QdrantClient(
            path=str(ROOT / "data/processed/qdrant")
        )
        self.embedder = TextEmbedding(model_name=MODEL_NAME)

    def close(self):
        self.client.close()

    def ingest(self):
        path = ROOT / "knowledge/concepts.jsonl"
        notes = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        if not notes:
            raise ValueError("Concept collection is empty.")

        ids = [note["concept_id"] for note in notes]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate concept IDs.")

        vectors = list(
            self.embedder.embed([note["text"] for note in notes])
        )

        if not self.client.collection_exists(COLLECTION):
            self.client.create_collection(
                collection_name=COLLECTION,
                vectors_config=models.VectorParams(
                    size=len(vectors[0]),
                    distance=models.Distance.COSINE,
                ),
            )

        points = [
            models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, note["concept_id"])),
                vector=vector.tolist(),
                payload={
                    **note,
                    "embedding_model": MODEL_NAME,
                },
            )
            for note, vector in zip(notes, vectors)
        ]

        self.client.upsert(
            collection_name=COLLECTION,
            points=points,
            wait=True,
        )
        return len(points)

    def retrieve(self, query, limit=2):
        if not query.strip():
            raise ValueError("Retrieval query must contain text.")

        if not self.client.collection_exists(COLLECTION):
            raise RuntimeError("Run the ingestion script first.")

        vector = next(self.embedder.query_embed(query))

        result = self.client.query_points(
            collection_name=COLLECTION,
            query=vector.tolist(),
            limit=limit,
            with_payload=True,
        )

        return [
            {
                **(point.payload or {}),
                "score": point.score,
            }
            for point in result.points
        ]
