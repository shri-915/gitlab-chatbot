def check_on_topic(query: str) -> bool:
    """Check if a user query is on-topic for the GitLab Handbook Assistant."""
    from rag.topic_rules import is_on_topic

    return is_on_topic(query)


def get_off_topic_response() -> str:
    """Return the standard off-topic rejection message."""
    return OFF_TOPIC_RESPONSE
