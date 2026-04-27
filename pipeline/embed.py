"""
Embedding Logic
================
Embeds text chunks locally using a Sentence-Transformers model.

Usage:
    from pipeline.embed import embed_chunks
    embedded = embed_chunks(chunks)
"""

import os
import time

from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOTENV_FILE = os.path.join(PROJECT_ROOT, ".env")

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=DOTENV_FILE)
except ImportError:
    pass

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "100"))
EMBEDDING_DIMENSION = int(os.environ.get("EMBEDDING_DIMENSION", "768"))
EMBEDDING_LOCAL_ENCODE_BATCH_SIZE = int(os.environ.get("EMBEDDING_LOCAL_ENCODE_BATCH_SIZE", "64"))
EMBEDDING_BATCH_CHAR_BUDGET = int(os.environ.get("EMBEDDING_BATCH_CHAR_BUDGET", "300000"))
EMBEDDING_BETWEEN_BATCH_DELAY = float(os.environ.get("EMBEDDING_BETWEEN_BATCH_DELAY", "0.0"))
EMBEDDING_RETRY_MIN_DELAY = int(os.environ.get("EMBEDDING_RETRY_MIN_DELAY", "3"))
EMBEDDING_BATCH_MAX_ATTEMPTS = int(os.environ.get("EMBEDDING_BATCH_MAX_ATTEMPTS", "3"))


# ---------------------------------------------------------------------------
# Model factory (singleton)
# ---------------------------------------------------------------------------
_model: SentenceTransformer = None


def _retry_non_value_errors(exc: Exception) -> bool:
    """Retry transient failures, but fail fast on configuration errors."""
    return not isinstance(exc, ValueError)


def _get_model() -> SentenceTransformer:
    """Get or create the local embedding model singleton."""
    global _model
    if _model is None:
        try:
            print(f"Loading local embedding model: {EMBEDDING_MODEL}")
            _model = SentenceTransformer(EMBEDDING_MODEL)
        except Exception as e:
            raise ValueError(
                f"Failed to load local embedding model '{EMBEDDING_MODEL}'. "
                "Ensure sentence-transformers is installed and model name is valid."
            ) from e

        model_dim_getter = getattr(_model, "get_embedding_dimension", None)
        if callable(model_dim_getter):
            model_dim = model_dim_getter()
        else:
            model_dim = _model.get_sentence_embedding_dimension()
        if model_dim and model_dim != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Embedding dimension mismatch: model '{EMBEDDING_MODEL}' outputs {model_dim} dims, "
                f"but EMBEDDING_DIMENSION is {EMBEDDING_DIMENSION}. "
                "Update EMBEDDING_DIMENSION or Supabase VECTOR dimension/schema."
            )

    return _model


def _iter_embedding_batches(chunks: list[dict]) -> list[list[dict]]:
    """Split chunks into batches that fit configured item and character budgets."""
    batches = []
    batch = []
    batch_chars = 0

    for chunk in chunks:
        chunk_chars = len(chunk.get("chunk_text", ""))
        if batch and (
            len(batch) >= BATCH_SIZE or batch_chars + chunk_chars > EMBEDDING_BATCH_CHAR_BUDGET
        ):
            batches.append(batch)
            batch = []
            batch_chars = 0

        batch.append(chunk)
        batch_chars += chunk_chars

    if batch:
        batches.append(batch)

    return batches


# ---------------------------------------------------------------------------
# Retry-wrapped embedding call
# ---------------------------------------------------------------------------
@retry(
    stop=stop_after_attempt(EMBEDDING_BATCH_MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=EMBEDDING_RETRY_MIN_DELAY, max=180),
    retry=retry_if_exception(_retry_non_value_errors),
    before_sleep=lambda rs: print(
        f"  ⚠ Embedding batch failed, retrying in {rs.next_action.sleep:.0f}s... "
        f"(attempt {rs.attempt_number})"
    ),
)
def _embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts using a local Sentence-Transformers model.
    Returns a list of embedding vectors (list of floats).
    """
    model = _get_model()
    vectors = model.encode(
        texts,
        batch_size=EMBEDDING_LOCAL_ENCODE_BATCH_SIZE,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return vectors.tolist()


@retry(
    stop=stop_after_attempt(EMBEDDING_BATCH_MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=EMBEDDING_RETRY_MIN_DELAY, max=180),
    retry=retry_if_exception(_retry_non_value_errors),
)
def embed_query(query: str) -> list[float]:
    """
    Embed a single query string for retrieval.
    """
    model = _get_model()
    vector = model.encode(
        [query],
        batch_size=1,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return vector[0].tolist()


# ---------------------------------------------------------------------------
# Main embedding function
# ---------------------------------------------------------------------------
def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed all chunks in batches of BATCH_SIZE.
    Adds an 'embedding' key to each chunk dict.

    Input: list of chunk dicts (must have 'chunk_text' key)
    Output: same list with 'embedding' key added to each
    """
    total = len(chunks)
    print(f"\nEmbedding {total} chunks in batches of {BATCH_SIZE}...")
    failed_batches = 0
    batch_start = 0

    for batch in _iter_embedding_batches(chunks):
        batch_end = batch_start + len(batch)
        texts = [c["chunk_text"] for c in batch]

        try:
            embeddings = _embed_batch(texts)
            for i, embedding in enumerate(embeddings):
                batch[i]["embedding"] = embedding

            progress = batch_end / total * 100
            print(f"  [{batch_end:>5}/{total}] ({progress:.1f}%) Embedded batch {batch_start}-{batch_end}")

        except Exception as e:
            failed_batches += 1
            print(f"  ✗ Failed to embed batch {batch_start}-{batch_end}: {e}")

        # Fixed cooldown between full-size batches to reduce quota pressure.
        if batch_end < total:
            time.sleep(EMBEDDING_BETWEEN_BATCH_DELAY)

        batch_start = batch_end

    embedded_count = sum(
        1 for c in chunks
        if "embedding" in c and any(v != 0.0 for v in c["embedding"])
    )
    print(f"\n✓ Embedding complete: {embedded_count}/{total} chunks successfully embedded")

    if embedded_count == 0:
        raise RuntimeError(
            "Embedding failed for all chunks. Verify local model configuration and dependencies."
        )

    if failed_batches:
        print(
            f"  ⚠ {failed_batches} batch(es) failed during embedding; proceeding with partial results."
        )

    return chunks
