import asyncio
from collections import defaultdict
from collections.abc import Callable

import numpy as np

from aimakerspace.openai_utils.embedding import EmbeddingModel


def cosine_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the cosine similarity between two vectors."""
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)


def dot_product_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the dot product similarity between two vectors."""
    return np.dot(vector_a, vector_b)


def euclidean_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Negative L2 distance so that higher is better."""
    return -float(np.linalg.norm(vector_a - vector_b, ord=2))


def manhattan_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Negative L1 distance so that higher is better."""
    return -float(np.sum(np.abs(vector_a - vector_b)))


DISTANCE_METRICS: dict[str, Callable[[np.array, np.array], float]] = {
    "cosine": cosine_similarity,
    "euclidean": euclidean_similarity,
    "manhattan": manhattan_similarity,
    "dot_product": dot_product_similarity,
}


class VectorDatabase:
    def __init__(self, embedding_model: EmbeddingModel = None):
        self.vectors = defaultdict(np.array)
        self.embedding_model = embedding_model or EmbeddingModel()

    def insert(self, key: str, vector: np.array) -> None:
        self.vectors[key] = vector

    def search(
        self,
        query_vector: np.array,
        k: int,
        distance_measure: Callable[[np.array, np.array], float] = cosine_similarity,
    ) -> list[tuple[str, float]]:
        scores = [
            (key, distance_measure(query_vector, vector))
            for key, vector in self.vectors.items()
        ]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:k]

    def search_by_text(
        self,
        query_text: str,
        k: int,
        distance_measure: Callable = cosine_similarity,
        return_as_text: bool = False,
    ) -> list[tuple[str, float]]:
        query_vector = self.embedding_model.get_embedding(query_text)
        results = self.search(query_vector, k, distance_measure)
        return [result[0] for result in results] if return_as_text else results

    def retrieve_from_key(self, key: str) -> np.array:
        return self.vectors.get(key, None)

    async def abuild_from_list(self, list_of_text: list[str]) -> "VectorDatabase":
        embeddings = await self.embedding_model.async_get_embeddings(list_of_text)
        for text, embedding in zip(list_of_text, embeddings, strict=True):
            self.insert(text, np.array(embedding))
        return self

    def search_by_text_with_token_budget(
        self,
        query_text: str,
        k: int,
        count_tokens: Callable[[str], int],
        max_context_tokens: int = 10000,
        distance_measure: Callable = cosine_similarity,
        reserve_tokens: int = 0,
    ) -> tuple[list[str], int, list[tuple[str, float]]]:
        """
        Returns:
          - selected_texts: list of chunk strings within budget
          - used_tokens: total tokens used by selected_texts
          - scored_results: original scored results for inspection
        """

        scored_results = self.search_by_text(
            query_text, k=k, distance_measure=distance_measure
        )
        selected_texts: list[str] = []
        used_tokens = 0
        limit = max(0, max_context_tokens - reserve_tokens)

        for text, _ in scored_results:
            t = count_tokens(text)
            if used_tokens + t > limit:
                break
            selected_texts.append(text)
            used_tokens += t

        return selected_texts, used_tokens, scored_results


if __name__ == "__main__":
    list_of_text = [
        "I like to eat broccoli and bananas.",
        "I ate a banana and spinach smoothie for breakfast.",
        "Chinchillas and kittens are cute.",
        "My sister adopted a kitten yesterday.",
        "Look at this cute hamster munching on a piece of broccoli.",
    ]

    vector_db = VectorDatabase()
    vector_db = asyncio.run(vector_db.abuild_from_list(list_of_text))
    k = 2

    searched_vector = vector_db.search_by_text("I think fruit is awesome!", k=k)
    print(f"Closest {k} vector(s):", searched_vector)

    retrieved_vector = vector_db.retrieve_from_key(
        "I like to eat broccoli and bananas."
    )
    print("Retrieved vector:", retrieved_vector)

    relevant_texts = vector_db.search_by_text(
        "I think fruit is awesome!", k=k, return_as_text=True
    )
    print(f"Closest {k} text(s):", relevant_texts)
