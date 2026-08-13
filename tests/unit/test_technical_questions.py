from app.processing.technical_questions import (
    extract_technical_questions, smart_sentence_split,
)


def test_asr_text_without_question_marks_is_split_and_categorized():
    text = ("Welcome to the interview How do you optimize a PostgreSQL query "
            "Walk me through a Kubernetes deployment")
    assert smart_sentence_split(text) == (
        "Welcome to the interview", "How do you optimize a PostgreSQL query",
        "Walk me through a Kubernetes deployment")
    assert extract_technical_questions(text) == (
        ("How do you optimize a PostgreSQL query", "Database"),
        ("Walk me through a Kubernetes deployment", "DevOps"),
    )


def test_non_question_conversation_is_not_misclassified():
    assert extract_technical_questions("We deployed Docker successfully today") == ()


def test_semantic_fallback_is_lazy_and_explicit():
    calls = []
    result = extract_technical_questions(
        "Explain the tradeoffs in this unusual architecture",
        lambda sentence: calls.append(sentence) or True)
    assert result == (("Explain the tradeoffs in this unusual architecture", "General"),)
    assert len(calls) == 1
