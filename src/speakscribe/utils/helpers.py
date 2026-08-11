"""Small public-safe helper functions."""


def language_code(value: str | None) -> str | None:
    """Convert locale-style language values to engine-friendly base codes."""
    if value is None:
        return None
    return value.replace("_", "-").split("-", 1)[0].lower()
