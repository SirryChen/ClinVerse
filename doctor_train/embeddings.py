"""Encode retrieval intents through an embedding endpoint."""
from __future__ import annotations
import math
from openai import OpenAI


class EmbeddingEncoder:
    def __init__(self, provider: dict):
        self.model = provider['embedding_model']
        self.client = OpenAI(api_key=provider['api_key'], base_url=provider['base_url'], timeout=provider.get('timeout', 180))

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for start in range(0, len(texts), 64):
            batch = texts[start:start + 64]
            response = self.client.embeddings.create(model=self.model, input=batch)
            items = sorted(response.data, key=lambda item: item.index)
            if [item.index for item in items] != list(range(len(batch))):
                raise ValueError('Embedding response indices do not match the input')
            vectors.extend(normalize(item.embedding) for item in items)
        return vectors


def normalize(vector: list[float]) -> list[float]:
    if not vector or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vector):
        raise ValueError('Embeddings require finite numeric vectors')
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        raise ValueError('Embeddings must have a positive norm')
    return [v / norm for v in vector]
