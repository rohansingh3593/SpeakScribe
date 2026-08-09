from decoding_policy import hotwords, initial_prompt


VOCABULARY = ("PostgreSQL", "MongoDB", "Jenkins", "CPU", "RAM")


def prompt(**overrides):
    values = {
        "final": True, "sample_count": 32_000, "sample_rate": 16_000,
        "language_mode": "hi", "vocabulary": VOCABULARY, "context": "",
    }
    values.update(overrides)
    return initial_prompt(**values)


def test_pinned_hindi_does_not_receive_english_instruction_or_vocabulary_prompt():
    assert prompt() is None


def test_hindi_can_use_genuine_prior_transcript_context():
    assert prompt(context="पिछला वाक्य पूरा हुआ") == "पिछला वाक्य पूरा हुआ"


def test_auto_hinglish_mode_does_not_misuse_initial_prompt_for_vocabulary():
    assert prompt(language_mode="auto") is None


def test_english_uses_real_context_without_instructions_or_vocabulary_injection():
    value = prompt(language_mode="en", context="The deployment finished")
    assert value == "The deployment finished"
    assert "language" not in value.casefold()


def test_partials_never_receive_a_prompt():
    assert prompt(final=False, language_mode="auto") is None


def test_subsecond_finals_do_not_receive_a_prompt():
    assert prompt(sample_count=15_999, language_mode="auto") is None


def test_final_hindi_rejects_latin_hotwords_to_preserve_devanagari_output():
    assert prompt() is None
    assert hotwords(final=True, language_mode="hi", vocabulary=VOCABULARY) is None


def test_auto_hinglish_uses_dedicated_hotwords_without_an_initial_prompt():
    assert prompt(language_mode="auto") is None
    assert hotwords(final=True, language_mode="auto", vocabulary=VOCABULARY) == (
        "PostgreSQL, MongoDB, Jenkins, CPU, RAM")


def test_partials_do_not_pay_for_hotword_bias():
    assert hotwords(final=False, language_mode="auto", vocabulary=VOCABULARY) is None
