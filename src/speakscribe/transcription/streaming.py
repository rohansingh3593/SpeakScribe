"""Small iterator helper shared by services and alternative frontends."""


def non_empty_results(results):
    for result in results:
        if result.text.strip():
            yield result
