"""The single, shared embedding model for OpsPilot.

Both ingestion (embedding documents) and retrieval (embedding the question)
import from here, so they always use the EXACT same model. Vectors from
different models aren't comparable, so this shared source prevents a subtle bug.
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from opspilot import config


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it (lru_cache = load on first call,
    return the same object every time after). Runs on the GPU per config."""
    return SentenceTransformer(config.EMBEDDING_MODEL, device=config.EMBEDDING_DEVICE)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Turn a list of texts into a list of embedding vectors (plain Python lists).

    normalize_embeddings=True scales every vector to length 1, which makes
    cosine-similarity comparisons clean and consistent.
    """
    model = get_embedding_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    return vectors.tolist()  # Chroma wants plain lists, not a NumPy array
