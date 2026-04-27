"""Local topic routing rules for GitLab handbook questions."""

from __future__ import annotations

import re


VALID_CATEGORIES = {
    "engineering",
    "people_ops",
    "product_direction",
    "values_culture",
    "meta",
    "general",
}

CATEGORY_BOOST_MAP = {
    "engineering": "handbook",
    "people_ops": "handbook",
    "product_direction": "direction",
    "values_culture": "handbook",
    "meta": None,
    "general": None,
}

CATEGORY_LABELS = {
    "engineering": "Engineering",
    "people_ops": "People Operations",
    "product_direction": "Product Direction",
    "values_culture": "Values & Culture",
    "meta": "About",
    "general": "General",
}

# Meta/identity keywords: questions about the chatbot itself or GitLab as a company
META_KEYWORDS = {
    "who are you",
    "what are you",
    "what do you do",
    "what can you do",
    "how do you work",
    "tell me about yourself",
    "introduce yourself",
    "what is this",
    "what is gitlab",
    "tell me about gitlab",
    "about gitlab",
    "what does gitlab do",
    "where is gitlab",
    "where is gitlab located",
    "gitlab headquarters",
    "gitlab hq",
    "gitlab history",
    "when was gitlab founded",
    "who founded gitlab",
    "gitlab ceo",
    "gitlab company",
    "gitlab employees",
    "is gitlab remote",
    "gitlab remote",
    "all remote",
    "all-remote",
    "gitlab public",
    "gitlab stock",
    "gitlab ipo",
    "how can you help",
    "help me",
    "what topics",
    "what questions",
}

ENGINEERING_KEYWORDS = {
    "engineering",
    "code review",
    "merge request",
    "mr",
    "deploy",
    "deployment",
    "ci",
    "cd",
    "pipeline",
    "incident",
    "postmortem",
    "infra",
    "infrastructure",
    "security",
    "architecture",
    "performance",
    "test",
    "testing",
    "feature flag",
    "rollback",
    "release",
}

PEOPLE_OPS_KEYWORDS = {
    "people ops",
    "hr",
    "hiring",
    "interview",
    "benefits",
    "compensation",
    "salary",
    "payroll",
    "pto",
    "vacation",
    "leave",
    "remote work",
    "work from anywhere",
    "performance review",
    "promotion",
    "onboarding",
    "manager",
    "recruiting",
}

PRODUCT_DIRECTION_KEYWORDS = {
    "product",
    "product direction",
    "roadmap",
    "strategy",
    "vision",
    "priorities",
    "what is gitlab building",
    "features",
    "direction",
}

VALUES_CULTURE_KEYWORDS = {
    "values",
    "culture",
    "credit",
    "handbook",
    "communication",
    "transparency",
    "collaboration",
    "behavior",
    "policy",
}

OFF_TOPIC_KEYWORDS = {
    "python",
    "javascript",
    "coding help",
    "math",
    "calculus",
    "weather",
    "recipe",
    "movie",
    "music",
    "relationship",
    "personal advice",
    "sports",
    "politics",
    "stock price",
}


def _normalize(query: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]+", " ", query.lower())).strip()


def _score(text: str, keywords: set[str]) -> int:
    return sum(1 for keyword in keywords if keyword in text)


def classify_query(query: str) -> tuple[str, str | None]:
    """Classify a query locally without calling Gemini."""
    text = _normalize(query)

    # Meta / identity questions — check first (highest priority)
    if _score(text, META_KEYWORDS):
        return "meta", None

    scores = {
        "engineering": _score(text, ENGINEERING_KEYWORDS),
        "people_ops": _score(text, PEOPLE_OPS_KEYWORDS),
        "product_direction": _score(text, PRODUCT_DIRECTION_KEYWORDS),
        "values_culture": _score(text, VALUES_CULTURE_KEYWORDS),
    }

    if not text:
        return "general", None

    if scores["engineering"]:
        category = "engineering"
    elif scores["people_ops"]:
        category = "people_ops"
    elif scores["product_direction"]:
        category = "product_direction"
    elif scores["values_culture"]:
        category = "values_culture"
    elif "gitlab" in text:
        category = "general"
    else:
        category = "general"

    return category, CATEGORY_BOOST_MAP[category]


def is_on_topic(query: str) -> bool:
    """Keep questions that are clearly about GitLab handbook / direction or the assistant itself."""
    text = _normalize(query)

    # Meta questions (about the chatbot / GitLab company) are always on-topic
    if _score(text, META_KEYWORDS):
        return True

    if "gitlab" in text:
        if _score(text, OFF_TOPIC_KEYWORDS):
            return False
        return True

    category, _ = classify_query(query)
    if category not in ("general", "meta"):
        return True

    if _score(text, OFF_TOPIC_KEYWORDS):
        return False

    return False


def get_category_label(category: str) -> str:
    return CATEGORY_LABELS.get(category, "📋 General")