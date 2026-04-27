"""
Guardrail — on-topic classification for GitLab Handbook Assistant.
Delegates to lightweight keyword rules in topic_rules.py.
"""

from rag.topic_rules import is_on_topic, CATEGORY_LABELS  # noqa: F401

OFF_TOPIC_RESPONSE = (
    "<strong>This question appears to be outside the scope of what I can help with.</strong><br><br>"
    "I'm designed to answer questions strictly grounded in <em>GitLab's Handbook</em> and "
    "<em>Direction pages</em> — topics like GitLab's values, engineering practices, "
    "people operations, remote culture, and product strategy.<br><br>"
    "If you believe your question is related to GitLab, try rephrasing it to include "
    "specific GitLab context (e.g. 'How does GitLab handle code reviews?' instead of "
    "a generic engineering question)."
)


def check_on_topic(query: str) -> bool:
    """Check if a user query is on-topic for the GitLab Handbook Assistant."""
    return is_on_topic(query)


def get_off_topic_response() -> str:
    """Return the standard off-topic rejection message."""
    return OFF_TOPIC_RESPONSE
