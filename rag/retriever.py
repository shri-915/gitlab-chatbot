"""
Supabase Vector Search Retriever
==================================
Handles embedding queries and performing similarity search against
the Supabase pgvector database using the match_chunks function.

Usage:
    from rag.retriever import retrieve_chunks
    results = retrieve_chunks(query, top_k=5)
"""

import os
import sys

from supabase import create_client, Client
from tenacity import retry, stop_after_attempt, wait_exponential

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.embed import embed_query


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_TOP_K = 5


# ---------------------------------------------------------------------------
# Supabase client
# ---------------------------------------------------------------------------
_supabase_client: Client = None


def _get_client() -> Client:
    """Get or create a Supabase client (singleton pattern)."""
    global _supabase_client
    if _supabase_client is None:
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_KEY", "")

        # Try Streamlit secrets as fallback
        if not url or not key:
            try:
                import streamlit as st
                url = url or st.secrets.get("SUPABASE_URL", "")
                key = key or st.secrets.get("SUPABASE_KEY", "")
            except Exception:
                pass

        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be configured.")

        _supabase_client = create_client(url, key)

    return _supabase_client


# ---------------------------------------------------------------------------
# Core retrieval functions
# ---------------------------------------------------------------------------
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=15),
)
def retrieve_chunks(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    filter_type: str = None,
) -> list[dict]:
    """
    Embed the query and perform similarity search in Supabase.

    Args:
        query: User's question
        top_k: Number of top results to return
        filter_type: Optional source_type filter ("handbook" or "direction")

    Returns:
        List of dicts with keys:
            chunk_text, source_url, section_title, page_title, source_type, similarity
    """
    # Step 1: Embed the query
    query_embedding = embed_query(query)

    # Step 2: Call the match_chunks Supabase function
    client = _get_client()

    params = {
        "query_embedding": query_embedding,
        "match_count": top_k,
    }
    if filter_type:
        params["filter_type"] = filter_type

    response = client.rpc("match_chunks", params).execute()

    results = response.data if response.data else []

    # Ensure consistent structure
    formatted_results = []
    for row in results:
        formatted_results.append({
            "id": row.get("id"),
            "chunk_text": row.get("chunk_text", ""),
            "source_url": row.get("source_url", ""),
            "section_title": row.get("section_title", ""),
            "page_title": row.get("page_title", ""),
            "source_type": row.get("source_type", ""),
            "similarity": row.get("similarity", 0.0),
        })

    return formatted_results


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=15),
)
def retrieve_chunks_boosted(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    boost_type: str = None,
    boost_factor: float = 0.05,
) -> list[dict]:
    """
    Embed the query and perform boosted similarity search in Supabase.
    Chunks matching boost_type get a similarity score bonus.

    Args:
        query: User's question
        top_k: Number of top results to return
        boost_type: source_type to boost ("handbook" or "direction")
        boost_factor: How much to boost matching chunks (default 0.05)

    Returns:
        Same format as retrieve_chunks
    """
    query_embedding = embed_query(query)

    client = _get_client()

    params = {
        "query_embedding": query_embedding,
        "match_count": top_k,
    }
    if boost_type:
        params["boost_type"] = boost_type
        params["boost_factor"] = boost_factor

    response = client.rpc("match_chunks_boosted", params).execute()

    results = response.data if response.data else []

    formatted_results = []
    for row in results:
        formatted_results.append({
            "id": row.get("id"),
            "chunk_text": row.get("chunk_text", ""),
            "source_url": row.get("source_url", ""),
            "section_title": row.get("section_title", ""),
            "page_title": row.get("page_title", ""),
            "source_type": row.get("source_type", ""),
            "similarity": row.get("similarity", 0.0),
        })

    return formatted_results


def get_top_similarity(results: list[dict]) -> float:
    """Get the highest similarity score from results."""
    if not results:
        return 0.0
    return max(r.get("similarity", 0.0) for r in results)
