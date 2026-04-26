"""
Section-Aware Query Router
============================
Classifies user queries into categories to enable section-aware
retrieval boosting. Uses the new google.genai SDK.

Usage:
    from rag.router import route_query
    category, boost_type = route_query("How does code review work?")
"""

import os

from google import genai
from google.genai import types as genai_types
from tenacity import retry, stop_after_attempt, wait_exponential


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROUTER_MODEL = os.environ.get("ROUTER_MODEL", "gemini-flash-latest")

VALID_CATEGORIES = {
    "engineering",
    "people_ops",
    "product_direction",
    "values_culture",
    "general",
}

# Map query categories → source_type boost for retrieval
CATEGORY_BOOST_MAP = {
    "engineering": "handbook",
    "people_ops": "handbook",
    "product_direction": "direction",
    "values_culture": "handbook",
    "general": None,
}

ROUTER_PROMPT = """Classify this question into exactly one category. Categories: engineering, people_ops, product_direction, values_culture, general

engineering: questions about GitLab's engineering processes, code review, deployment, technical practices, development workflows
people_ops: questions about HR, hiring, benefits, remote work policies, performance reviews, team structure, compensation
product_direction: questions about GitLab's product roadmap, features, strategy, what GitLab is building
values_culture: questions about GitLab's core values, CREDIT values, how GitLab operates as a company, communication norms
general: anything that doesn't clearly fit the above

Respond with ONLY the category name, nothing else.

Question: {query}"""


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
            http_options=genai_types.HttpOptions(api_version="v1beta"),
        )
    return _client


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def route_query(query: str) -> tuple[str, str | None]:
    """
    Classify a user query into a category and determine retrieval boost.

    Returns:
        Tuple of (category, boost_type):
            - category: one of the VALID_CATEGORIES strings
            - boost_type: "handbook", "direction", or None
    """
    client = _get_client()
    prompt = ROUTER_PROMPT.format(query=query)

    try:
        response = client.models.generate_content(
            model=ROUTER_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=10,
            ),
        )

        raw_text = (response.text or "").strip()
        if not raw_text:
            return "general", None

        result = raw_text.lower().replace(" ", "_")

        if result in VALID_CATEGORIES:
            category = result
        else:
            # Fuzzy match
            for valid_cat in VALID_CATEGORIES:
                if valid_cat in result or result in valid_cat:
                    category = valid_cat
                    break
            else:
                category = "general"

        boost_type = CATEGORY_BOOST_MAP.get(category)
        return category, boost_type

    except Exception as e:
        print(f"  ⚠ Router error: {e} — defaulting to 'general'")
        return "general", None


def get_category_label(category: str) -> str:
    """Return a human-readable label for a category."""
    labels = {
        "engineering": "🔧 Engineering",
        "people_ops": "👥 People Operations",
        "product_direction": "🚀 Product Direction",
        "values_culture": "💎 Values & Culture",
        "general": "📋 General",
    }
    return labels.get(category, "📋 General")
