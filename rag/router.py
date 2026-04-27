def route_query(query: str) -> tuple[str, str | None]:
    """Classify a user query into a category and determine retrieval boost."""
    from rag.topic_rules import classify_query

    return classify_query(query)


from rag.topic_rules import get_category_label
