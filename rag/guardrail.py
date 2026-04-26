"""
On-Topic Guardrail Classifier
================================
Uses a lightweight Gemini prompt to classify whether a user's question
is on-topic (related to GitLab's handbook/direction) or off-topic.
Uses the new google.genai SDK.

Usage:
    from rag.guardrail import check_on_topic
    is_on_topic = check_on_topic("What are GitLab's values?")
"""

import os

from google import genai
from google.genai import types as genai_types
from tenacity import retry, stop_after_attempt, wait_exponential


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CLASSIFIER_MODEL = "gemini-1.5-flash"

GUARDRAIL_PROMPT = """You are a strict topic classifier. A user is interacting with a chatbot that ONLY answers questions about GitLab's internal handbook, company culture, values, engineering practices, people operations, product direction, and work processes.

Classify the following user message as either ON_TOPIC or OFF_TOPIC.

ON_TOPIC means: the question is genuinely about GitLab as a company — its policies, processes, values, product areas, how teams work, remote work practices, career paths at GitLab, engineering culture, or product direction.

OFF_TOPIC means: the question is about something unrelated to GitLab's handbook or direction — general coding help, current events, personal advice, other companies, creative writing, math, etc.

Respond with ONLY one word: ON_TOPIC or OFF_TOPIC. No explanation.

User message: {query}"""

OFF_TOPIC_RESPONSE = (
    "I'm built specifically to answer questions about GitLab's Handbook and Direction. "
    "I can't help with that — but try asking me about GitLab's values, engineering "
    "processes, product roadmap, or team policies."
)


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
# Classifier
# ---------------------------------------------------------------------------
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def check_on_topic(query: str) -> bool:
    """
    Check if a user query is on-topic for the GitLab Handbook Assistant.

    Returns:
        True if on-topic, False if off-topic
    """
    client = _get_client()
    prompt = GUARDRAIL_PROMPT.format(query=query)

    try:
        response = client.models.generate_content(
            model=CLASSIFIER_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=10,
            ),
        )

        result = response.text.strip().upper()

        if "ON_TOPIC" in result:
            return True
        elif "OFF_TOPIC" in result:
            return False
        else:
            # Fail open — don't block legitimate questions
            print(f"  ⚠ Guardrail unexpected result: '{result}' — defaulting to ON_TOPIC")
            return True

    except Exception as e:
        # Fail open on errors
        print(f"  ⚠ Guardrail error: {e} — defaulting to ON_TOPIC")
        return True


def get_off_topic_response() -> str:
    """Return the standard off-topic rejection message."""
    return OFF_TOPIC_RESPONSE
