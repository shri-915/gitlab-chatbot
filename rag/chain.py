"""
Full RAG Chain
================
Orchestrates the complete RAG pipeline:
    Guardrail → Route → Retrieve → Confidence Check → Generate → Format
Uses the new google.genai SDK.

Usage:
    from rag.chain import run_rag_chain
    result = run_rag_chain(query="What are GitLab's values?", history=[])
"""

import os

from google import genai
from google.genai import types as genai_types
from tenacity import retry, stop_after_attempt, wait_exponential

from rag.guardrail import check_on_topic, get_off_topic_response
from rag.router import route_query, get_category_label
from rag.retriever import retrieve_chunks, retrieve_chunks_boosted, get_top_similarity


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LLM_MODEL = "gemini-1.5-flash"
CONFIDENCE_THRESHOLD = 0.75
TOP_K = 5

SYSTEM_PROMPT = """You are GitLab's official Handbook Assistant. You help GitLab employees and candidates understand GitLab's culture, processes, values, and product direction.

STRICT RULES:
1. Answer ONLY using the provided context chunks from GitLab's Handbook and Direction pages. Do not use any outside knowledge.
2. If the context does not contain enough information to answer confidently, say so explicitly. Do not guess or fill gaps with general knowledge.
3. Be direct and specific. Quote exact policy language when relevant.
4. At the end of your answer, list the sources you used as "Sources Used:" followed by the section titles (the source URLs will be added by the system).
5. If asked about something not covered in the provided context, say: "The provided handbook sections don't cover this specifically. I'd recommend checking [relevant section] directly."

Context from GitLab Handbook/Direction:
{context}

Conversation history:
{history}

User question: {query}"""


# ---------------------------------------------------------------------------
# Client factory (singleton)
# ---------------------------------------------------------------------------
_client: genai.Client = None


def _get_client() -> genai.Client:
    """Get or create a google.genai Client."""
    global _client
    if _client is None:
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            try:
                import streamlit as st
                api_key = st.secrets.get("GOOGLE_API_KEY", "")
            except Exception:
                pass
        if not api_key:
            try:
                from dotenv import load_dotenv
                load_dotenv()
                api_key = os.environ.get("GOOGLE_API_KEY", "")
            except ImportError:
                pass
        _client = genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(api_version="v1"),
        )
    return _client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context string for the LLM."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Source {i}]\n"
            f"Section: {chunk['section_title']}\n"
            f"Page: {chunk['page_title']}\n"
            f"Type: {chunk['source_type']}\n"
            f"Content:\n{chunk['chunk_text']}\n"
        )
    return "\n---\n".join(parts)


def _format_history(history: list[dict], max_turns: int = 3) -> str:
    """Format last N conversation turns for the prompt."""
    if not history:
        return "No previous conversation."
    recent = history[-(max_turns * 2):]
    formatted = []
    for msg in recent:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if len(content) > 500:
            content = content[:500] + "..."
        formatted.append(f"{role.upper()}: {content}")
    return "\n".join(formatted)


def _format_sources(chunks: list[dict]) -> list[dict]:
    """Deduplicate and format source info for display."""
    seen_urls = set()
    sources = []
    for chunk in chunks:
        url = chunk.get("source_url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            sources.append({
                "section_title": chunk.get("section_title", "Unknown Section"),
                "source_url": url,
                "page_title": chunk.get("page_title", ""),
                "source_type": chunk.get("source_type", ""),
            })
    return sources


# ---------------------------------------------------------------------------
# LLM Generation
# ---------------------------------------------------------------------------
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=15),
)
def _generate_answer(prompt: str) -> str:
    """Call Gemini via the new google.genai SDK to generate an answer."""
    client = _get_client()
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=1500,
            top_p=0.9,
        ),
    )
    return response.text


# ---------------------------------------------------------------------------
# Main RAG Chain
# ---------------------------------------------------------------------------
def run_rag_chain(query: str, history: list[dict] = None) -> dict:
    """
    Execute the full RAG chain for a user query.

    Args:
        query: User's question
        history: Conversation history [{"role": "user"/"assistant", "content": "..."}]

    Returns:
        Dict with keys:
            answer, sources, chunks, is_on_topic, is_confident,
            confidence_score, category, category_label, error
    """
    history = history or []

    result = {
        "answer": "",
        "sources": [],
        "chunks": [],
        "is_on_topic": True,
        "is_confident": True,
        "confidence_score": 0.0,
        "category": "general",
        "category_label": "📋 General",
        "error": None,
    }

    try:
        # ── Step 1: Guardrail ─────────────────────────────────────────────
        is_on_topic = check_on_topic(query)
        result["is_on_topic"] = is_on_topic
        if not is_on_topic:
            result["answer"] = get_off_topic_response()
            return result

        # ── Step 2: Route ─────────────────────────────────────────────────
        category, boost_type = route_query(query)
        result["category"] = category
        result["category_label"] = get_category_label(category)

        # ── Step 3: Retrieve ──────────────────────────────────────────────
        if boost_type:
            chunks = retrieve_chunks_boosted(
                query=query,
                top_k=TOP_K,
                boost_type=boost_type,
                boost_factor=0.05,
            )
        else:
            chunks = retrieve_chunks(query=query, top_k=TOP_K)

        result["chunks"] = chunks

        # ── Step 4: Confidence Check ──────────────────────────────────────
        top_score = get_top_similarity(chunks)
        result["confidence_score"] = top_score
        result["is_confident"] = top_score >= CONFIDENCE_THRESHOLD

        # ── Step 5: Generate ──────────────────────────────────────────────
        context = _format_context(chunks)
        history_str = _format_history(history)
        prompt = SYSTEM_PROMPT.format(
            context=context,
            history=history_str,
            query=query,
        )
        answer = _generate_answer(prompt)

        # ── Step 6: Format ────────────────────────────────────────────────
        sources = _format_sources(chunks)
        result["sources"] = sources

        if result["is_confident"]:
            result["answer"] = answer
        else:
            result["answer"] = (
                "⚠️ I found some related content but I'm not confident it fully answers "
                "your question. Here's what I found:\n\n"
                f"{answer}\n\n"
                "Please verify at the sources listed below."
            )

        return result

    except Exception as e:
        result["error"] = str(e)
        result["answer"] = (
            "I'm sorry, I encountered an error processing your question. "
            "Please try again in a moment."
        )
        print(f"  ✗ RAG chain error: {e}")
        return result
