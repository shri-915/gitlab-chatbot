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
from google.genai.errors import ClientError
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential, retry_if_exception

from gemini_limits import acquire, estimate_tokens, scale_quotas_for_key_count
from rag.guardrail import check_on_topic, get_off_topic_response
from rag.key_pool import GeminiKeyPool
from rag.router import route_query, get_category_label
from rag.retriever import retrieve_chunks, retrieve_chunks_boosted, get_top_similarity
from rag.topic_rules import META_KEYWORDS, _normalize, _score


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-flash-latest")
CONFIDENCE_THRESHOLD = 0.75
TOP_K = 5


def _retry_transient_generation_errors(exc: Exception) -> bool:
    """Retry only transient model errors; rotate API key on rate-limit."""
    if isinstance(exc, ClientError):
        code = getattr(exc, "code", None)
        if code == 429:
            pool = _get_pool()
            current = pool.current_key()
            pool.mark_rate_limited(current, backoff_seconds=62.0)
            if pool.key_count > 1:
                new_key = pool.rotate()
                print(f"Rate limited on key …{current[-6:]} — rotating to …{new_key[-6:]}")
            return True
        return isinstance(code, int) and code >= 500
    return True

SYSTEM_PROMPT = """You are the GitLab Handbook Assistant — an AI built to help GitLab team members, candidates, and the public understand GitLab's culture, processes, values, and product direction.

## About You
You are an AI-powered RAG (Retrieval-Augmented Generation) chatbot grounded strictly in GitLab's official Handbook (handbook.gitlab.com) and Direction pages (about.gitlab.com/direction). Your answers always cite the source sections used. You were built as a demonstration of how AI can make GitLab's handbook more accessible.

## About GitLab (company facts you may use)
- GitLab is an open-core, all-remote DevSecOps platform company headquartered in San Francisco, California, USA (although all-remote means team members are distributed across 60+ countries).
- Founded in 2011 by Dmitriy Zaporozhets and Valery Sizov. CEO is Sid Sijbrandij.
- GitLab went public (IPO) on NASDAQ on October 14, 2021 under the ticker GTLB.
- GitLab is the world's largest all-remote company, with no physical offices.
- The GitLab product is a single DevSecOps platform covering the entire software development lifecycle: planning, SCM, CI/CD, security, and monitoring.
- GitLab's mission: "Everyone can contribute."
- GitLab has over 30 million registered users and is used by more than 50% of Fortune 100 companies.
- GitLab's six core values (CREDIT): Collaboration, Results, Efficiency, Diversity Inclusion & Belonging, Iteration, Transparency.

## What You Can Help With
You can answer questions about:
- GitLab's values, culture, and operating principles
- Engineering processes: code review, MRs, deployments, incident response
- People Operations: hiring, benefits, remote work, career growth
- Product direction, roadmap, and strategy
- GitLab as a company (history, structure, remote culture)
- Any topic covered in the official Handbook or Direction pages

## Strict Answer Rules
1. For questions about GitLab the company or yourself (the assistant), you may use the facts listed above without needing context chunks.
2. For all other questions, answer ONLY using the provided context chunks from GitLab's Handbook and Direction pages. Do not add outside knowledge.
3. If the context does not contain enough information to answer confidently, say so explicitly. Do not guess or fill gaps with general knowledge.
4. Be direct and specific. Quote exact policy language when relevant.
5. At the end of your answer, list the sources you used as "Sources Used:" followed by the section titles (the source URLs will be added by the system).
6. If asked about something not covered in the provided context, say: "The provided handbook sections don't cover this specifically. I'd recommend checking [relevant section] directly."

Context from GitLab Handbook/Direction:
{context}

Conversation history:
{history}

User question: {query}"""


# ---------------------------------------------------------------------------
# Meta / identity response (no API call needed)
# ---------------------------------------------------------------------------
META_RESPONSE = (
    "<strong>I'm the GitLab Handbook Assistant</strong> — an AI chatbot grounded strictly in "
    "GitLab's official <a href='https://handbook.gitlab.com/' target='_blank'>Handbook</a> and "
    "<a href='https://about.gitlab.com/direction/' target='_blank'>Direction</a> pages.<br><br>"
    "<strong>About me:</strong><br>"
    "I use Retrieval-Augmented Generation (RAG): your question is embedded, matched against "
    "thousands of handbook chunks stored in a Supabase vector database, and answered by "
    "Gemini Flash — with every answer citing the exact source sections used.<br><br>"
    "<strong>About GitLab:</strong><br>"
    "GitLab is an open-core, all-remote DevSecOps platform company founded in 2011 by "
    "Dmitriy Zaporozhets and Valery Sizov. It is headquartered in San Francisco, CA, USA — "
    "though as the <em>world's largest all-remote company</em>, its ~2,000+ team members "
    "are distributed across 60+ countries with no physical offices. "
    "GitLab went public on NASDAQ in October 2021 (ticker: GTLB) and serves over "
    "30 million registered users, including more than 50% of Fortune 100 companies. "
    "Its mission is <em>\"Everyone can contribute.\"</em><br><br>"
    "<strong>What I can help you with:</strong><br>"
    "<ul>"
    "<li>GitLab's values, culture &amp; operating principles (CREDIT)</li>"
    "<li>Engineering processes: code review, merge requests, CI/CD, incident response</li>"
    "<li>People Operations: hiring, benefits, remote work, career growth</li>"
    "<li>Product direction, roadmap &amp; strategy</li>"
    "<li>GitLab company background, history &amp; remote culture</li>"
    "</ul>"
    "Just ask your question and I'll find the most relevant handbook sections for you."
)


def _is_meta_query(query: str) -> bool:
    """Return True for chatbot-identity / GitLab-company questions."""
    return bool(_score(_normalize(query), META_KEYWORDS))


# ---------------------------------------------------------------------------
# Client factory (pool-backed singleton)
# ---------------------------------------------------------------------------
_pool: GeminiKeyPool | None = None
_clients: dict[str, genai.Client] = {}


def _get_pool() -> GeminiKeyPool:
    """Get or create the shared GeminiKeyPool (lazy, once per process)."""
    global _pool
    if _pool is None:
        _pool = GeminiKeyPool.from_env()
        scale_quotas_for_key_count(_pool.key_count)
    return _pool


def _get_client(api_key: str | None = None) -> genai.Client:
    """Get or create a google.genai Client for the given API key."""
    if api_key is None:
        api_key = _get_pool().current_key()
    if api_key not in _clients:
        _clients[api_key] = genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(api_version="v1beta"),
        )
    return _clients[api_key]


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
    retry=retry_if_exception(_retry_transient_generation_errors),
)
def _generate_answer(prompt: str) -> str:
    """Call Gemini via the google.genai SDK to generate an answer."""
    acquire("flash", estimate_tokens(prompt) + 1500, operation="answer generation")
    pool = _get_pool()
    api_key = pool.current_key()
    client = _get_client(api_key)
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

        # ── Step 1b: Meta / identity short-circuit (no API call needed) ───
        if _is_meta_query(query):
            result["category"] = "meta"
            result["category_label"] = "About"
            result["answer"] = META_RESPONSE
            result["is_confident"] = True
            result["confidence_score"] = 1.0
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
            result["answer"] = answer

        return result

    except Exception as e:
        root_error = e
        if isinstance(e, RetryError):
            try:
                root_error = e.last_attempt.exception() or e
            except Exception:
                root_error = e

        result["error"] = str(root_error)
        err_str = str(root_error).lower()

        if isinstance(root_error, ValueError):
            result["answer"] = (
                "<strong>Configuration error</strong><br><br>"
                f"There is a setup issue that prevented the assistant from loading: <em>{root_error}</em><br><br>"
                "Please check that GOOGLE_API_KEY, SUPABASE_URL, and SUPABASE_KEY are correctly configured."
            )
        elif "quota" in err_str or "429" in err_str or "rate" in err_str:
            result["answer"] = (
                "<strong>The assistant has reached its API rate limit.</strong><br><br>"
                "This is a temporary restriction on the free-tier API. "
                "Please wait a minute and try your question again."
            )
        elif "connect" in err_str or "network" in err_str or "timeout" in err_str:
            result["answer"] = (
                "<strong>Connection issue detected.</strong><br><br>"
                "The assistant couldn't reach the Gemini API or the Supabase database. "
                "Please check your internet connection and try again."
            )
        else:
            result["answer"] = (
                "<strong>Something went wrong while processing your question.</strong><br><br>"
                "This is likely a transient error. Please try again in a moment. "
                "If the problem persists, try rephrasing your question."
            )

        print(f"RAG chain error: {root_error}")
        return result
