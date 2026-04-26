"""
Embedding Logic
================
Embeds text chunks using Google's text-embedding-004 model via the new google-genai SDK.
Handles batching (groups of 20) and retry logic with tenacity.

Usage:
    from pipeline.embed import embed_chunks
    embedded = embed_chunks(chunks)
"""

import os
import time

from google import genai
from google.genai import types as genai_types
from google.genai.errors import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "gemini-embedding-001"
BATCH_SIZE = 20           # Google API recommended batch size
EMBEDDING_DIMENSION = 768  # Chosen to match Supabase VECTOR(768) schema


# ---------------------------------------------------------------------------
# Client factory (singleton)
# ---------------------------------------------------------------------------
_client: genai.Client = None


def _retry_non_value_errors(exc: Exception) -> bool:
    """Retry transient failures, but fail fast on configuration and client-side API errors."""
    if isinstance(exc, ClientError):
        code = getattr(exc, "code", None)
        # Retry only rate-limits and server-side failures.
        return code == 429 or (isinstance(code, int) and code >= 500)
    return not isinstance(exc, ValueError)


def _get_client() -> genai.Client:
    """Get or create a google.genai Client (singleton) using v1 API."""
    global _client
    if _client is None:
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            try:
                from dotenv import load_dotenv
                load_dotenv()
                api_key = os.environ.get("GOOGLE_API_KEY", "")
            except ImportError:
                pass
        if not api_key:
            try:
                import streamlit as st
                api_key = st.secrets.get("GOOGLE_API_KEY", "")
            except Exception:
                pass
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found. Set it in .env or as an environment variable."
            )
        _client = genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(api_version="v1beta"),
        )
    return _client


# ---------------------------------------------------------------------------
# Retry-wrapped embedding call
# ---------------------------------------------------------------------------
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception(_retry_non_value_errors),
    before_sleep=lambda rs: print(
        f"  ⚠ Embedding API call failed, retrying in {rs.next_action.sleep:.0f}s... "
        f"(attempt {rs.attempt_number})"
    ),
)
def _embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts using the new google.genai SDK.
    Returns a list of embedding vectors (list of floats).
    """
    client = _get_client()
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=genai_types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=EMBEDDING_DIMENSION,
        ),
    )
    # response.embeddings is a list of ContentEmbedding objects with .values
    return [emb.values for emb in response.embeddings]


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception(_retry_non_value_errors),
)
def embed_query(query: str) -> list[float]:
    """
    Embed a single query string for retrieval.
    Uses RETRIEVAL_QUERY task type for better search performance.
    """
    client = _get_client()
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[query],
        config=genai_types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=EMBEDDING_DIMENSION,
        ),
    )
    return response.embeddings[0].values


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

    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch = chunks[batch_start:batch_end]
        texts = [c["chunk_text"] for c in batch]

        try:
            embeddings = _embed_batch(texts)
            for i, embedding in enumerate(embeddings):
                chunks[batch_start + i]["embedding"] = embedding

            progress = batch_end / total * 100
            print(f"  [{batch_end:>5}/{total}] ({progress:.1f}%) Embedded batch {batch_start}-{batch_end}")

        except Exception as e:
            print(f"  ✗ Failed to embed batch {batch_start}-{batch_end}: {e}")
            failed_batches += 1

        # Polite delay between batches to respect rate limits
        if batch_end < total:
            time.sleep(0.3)

    embedded_count = sum(
        1 for c in chunks
        if "embedding" in c and any(v != 0.0 for v in c["embedding"])
    )
    print(f"\n✓ Embedding complete: {embedded_count}/{total} chunks successfully embedded")

    if embedded_count == 0:
        raise RuntimeError(
            "Embedding failed for all chunks. Verify GOOGLE_API_KEY and model availability."
        )

    if failed_batches:
        print(
            f"  ⚠ {failed_batches} batch(es) failed during embedding; proceeding with partial results."
        )

    return chunks
