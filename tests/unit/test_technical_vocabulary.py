"""Coverage for the expanded default vocabulary and lightweight topic matching."""

from app.config.settings import DEFAULT_VOCABULARY
from app.config.technical_vocabulary import detect_technical_topics


def test_default_vocabulary_contains_requested_domains_and_terms():
    for term in ("cpython", "django rest framework", "elastic kubernetes service",
                 "rate limit header", "materialized view", "react router",
                 "circuit breaker", "github actions", "bankers algorithm",
                 "boundary value analysis", "azure sentinel", "xgboost"):
        assert term in DEFAULT_VOCABULARY


def test_vocabulary_is_deduplicated_without_losing_canonical_core_terms():
    assert len(DEFAULT_VOCABULARY) == len(set(DEFAULT_VOCABULARY))
    assert "SQLAlchemy" in DEFAULT_VOCABULARY


def test_topic_detection_uses_words_not_substrings():
    topics = detect_technical_topics(
        "How do you deploy a Django REST Framework API with Kubernetes?")
    assert {"Django", "API", "DevOps"} <= set(topics)
    assert detect_technical_topics("We will reactivate the account") == ()

