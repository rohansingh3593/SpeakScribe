"""Reusable prerecorded-audio accuracy and latency evaluation for SpeakScribe."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import time
import wave


TECHNICAL_TERMS = {
    "python", "pyqt6", "sqlalchemy", "alembic", "fastapi", "pydantic",
    "jenkins", "docker", "kubernetes", "git", "github", "gitlab", "jira",
    "api", "pr", "pytest", "postgresql", "mongodb", "kafka", "redis",
}


@dataclass(frozen=True)
class EditDetails:
    missing: list[str]
    extra: list[str]
    substitutions: list[str]


@dataclass
class EvaluationResult:
    case_id: str
    audio: str
    language: str
    expected: str
    actual: str
    detected_language: str
    similarity: float
    wer: float
    status: str
    duration_seconds: float
    inference_seconds: float
    real_time_factor: float
    missing_words: list[str]
    extra_words: list[str]
    substitutions: list[str]
    duplicated_words: list[str]
    technical_term_problems: list[str]
    possible_problem: str


def normalize_transcript(text: str) -> str:
    """Normalize comparison-only differences without changing word meaning."""
    text = text.casefold().replace("।", " ")
    text = re.sub(r"[^\w\u0900-\u097f']+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _alignment(reference: list[str], hypothesis: list[str]) -> tuple[int, EditDetails]:
    rows, columns = len(reference) + 1, len(hypothesis) + 1
    costs = [[0] * columns for _ in range(rows)]
    steps = [[""] * columns for _ in range(rows)]
    for row in range(1, rows):
        costs[row][0], steps[row][0] = row, "delete"
    for column in range(1, columns):
        costs[0][column], steps[0][column] = column, "insert"
    for row in range(1, rows):
        for column in range(1, columns):
            choices = [
                (costs[row - 1][column] + 1, "delete"),
                (costs[row][column - 1] + 1, "insert"),
                (costs[row - 1][column - 1] +
                 (reference[row - 1] != hypothesis[column - 1]),
                 "equal" if reference[row - 1] == hypothesis[column - 1] else "replace"),
            ]
            costs[row][column], steps[row][column] = min(choices, key=lambda item: item[0])

    missing, extra, substitutions = [], [], []
    row, column = len(reference), len(hypothesis)
    while row or column:
        operation = steps[row][column]
        if operation == "equal":
            row, column = row - 1, column - 1
        elif operation == "replace":
            substitutions.append(f"{reference[row - 1]} -> {hypothesis[column - 1]}")
            row, column = row - 1, column - 1
        elif operation == "delete":
            missing.append(reference[row - 1])
            row -= 1
        else:
            extra.append(hypothesis[column - 1])
            column -= 1
    return costs[-1][-1], EditDetails(missing[::-1], extra[::-1], substitutions[::-1])


def compare_transcripts(expected: str, actual: str) -> tuple[float, float, EditDetails]:
    reference = normalize_transcript(expected).split()
    hypothesis = normalize_transcript(actual).split()
    edits, details = _alignment(reference, hypothesis)
    wer = edits / max(1, len(reference))
    word_accuracy = 1.0 - min(1.0, wer)
    sequence = SequenceMatcher(None, reference, hypothesis, autojunk=False).ratio()
    similarity = 100.0 * (0.7 * word_accuracy + 0.3 * sequence)
    return round(similarity, 2), round(wer, 4), details


def status_for_similarity(similarity: float) -> str:
    if similarity >= 80:
        return "PASS"
    if similarity >= 60:
        return "PASS WITH WARNING"
    return "FAIL"


def load_wav(path: Path, target_rate: int):
    """Load PCM WAV with stdlib wave; import NumPy only for real evaluation."""
    import numpy as np
    from audio_pipeline import resample_audio_block

    with wave.open(str(path), "rb") as stream:
        channels = stream.getnchannels()
        sample_width = stream.getsampwidth()
        source_rate = stream.getframerate()
        frames = stream.readframes(stream.getnframes())
    if sample_width not in (1, 2, 4):
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[sample_width]
    audio = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    if sample_width == 1:
        audio = (audio - 128.0) / 128.0
    else:
        audio /= float(2 ** (sample_width * 8 - 1))
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return resample_audio_block(audio, source_rate, target_rate)


def _diagnosis(result: EvaluationResult) -> str:
    if result.detected_language.casefold() != result.language.casefold() and result.language != "Hinglish":
        return "Language detection or language-mode selection mismatch."
    if result.technical_term_problems:
        return "Technical vocabulary recognition/context prompting needs review."
    if result.extra_words and len(result.extra_words) > len(result.missing_words) + 2:
        return "Possible background-noise hallucination or overly permissive speech detection."
    if result.missing_words:
        return "Possible clipping, silence segmentation, low input level, or decoding omission."
    return "Review pronunciation, model profile, and conservative text cleanup."


def evaluate_case(case: dict, root: Path, provider) -> EvaluationResult:
    from audio_pipeline import ASRJob
    from config import AppConfig, PerformanceMode

    path = root / case["audio"]
    language_hint = {"English": "en", "Hindi": "hi", "Hinglish": "hi"}[case["language"]]
    config = AppConfig(
        language_mode=language_hint,
        performance_mode=PerformanceMode(case.get("performance", "balanced")),
    )
    audio = load_wav(path, config.sample_rate)
    job = ASRJob(audio=audio, final=True, utterance_id=1, captured_at=time.monotonic())
    started = time.monotonic()
    actual, detected = provider.get(config).transcribe(job, "")
    inference = time.monotonic() - started
    duration = len(audio) / config.sample_rate
    similarity, wer, edits = compare_transcripts(case["expected"], actual)
    expected_terms = TECHNICAL_TERMS.intersection(normalize_transcript(case["expected"]).split())
    actual_terms = set(normalize_transcript(actual).split())
    technical_problems = sorted(expected_terms - actual_terms)
    expected_counts = Counter(normalize_transcript(case["expected"]).split())
    actual_counts = Counter(normalize_transcript(actual).split())
    duplicated = sorted(word for word, count in actual_counts.items()
                        if count > max(1, expected_counts[word]) + 1)
    result = EvaluationResult(
        case_id=case["id"], audio=case["audio"], language=case["language"],
        expected=case["expected"], actual=actual, detected_language=detected,
        similarity=similarity, wer=wer, status=status_for_similarity(similarity),
        duration_seconds=round(duration, 3), inference_seconds=round(inference, 3),
        real_time_factor=round(inference / max(duration, .001), 3),
        missing_words=edits.missing, extra_words=edits.extra,
        substitutions=edits.substitutions,
        duplicated_words=duplicated,
        technical_term_problems=technical_problems, possible_problem="",
    )
    result.possible_problem = _diagnosis(result)
    return result


def render_markdown(results: list[EvaluationResult], missing: list[str]) -> str:
    counts = defaultdict(int)
    by_language: dict[str, list[EvaluationResult]] = defaultdict(list)
    for result in results:
        counts[result.status] += 1
        by_language[result.language].append(result)
    lines = ["# Speech Recognition Test Report", "",
             f"- Total configured: {len(results) + len(missing)}",
             f"- Executed: {len(results)}", f"- Missing audio: {len(missing)}",
             f"- Passed: {counts['PASS']}",
             f"- Passed with warning: {counts['PASS WITH WARNING']}",
             f"- Failed: {counts['FAIL']}", ""]
    if results:
        accepted = counts["PASS"] + counts["PASS WITH WARNING"]
        lines += [f"- Overall pass rate: {100 * accepted / len(results):.1f}%",
                  f"- Average similarity: "
                  f"{sum(item.similarity for item in results) / len(results):.1f}%",
                  f"- Average inference: "
                  f"{sum(item.inference_seconds for item in results) / len(results):.3f}s",
                  f"- Average RTF: "
                  f"{sum(item.real_time_factor for item in results) / len(results):.3f}", ""]
    for language in ("English", "Hindi", "Hinglish"):
        items = by_language[language]
        if items:
            lines.append(f"- {language} average similarity: "
                         f"{sum(item.similarity for item in items) / len(items):.1f}%")
            lines.append(f"- {language} average WER: "
                         f"{sum(item.wer for item in items) / len(items):.3f}")
    lines += ["", "| Test | Language | Similarity | WER | Inference | RTF | Status |",
              "|---|---|---:|---:|---:|---:|---|"]
    for item in results:
        lines.append(f"| {item.audio} | {item.language} | {item.similarity:.1f}% | "
                     f"{item.wer:.3f} | {item.inference_seconds:.2f}s | "
                     f"{item.real_time_factor:.2f} | {item.status} |")
    for item in results:
        if item.status == "PASS":
            continue
        lines += ["", f"## {item.case_id}: {item.status}", "",
                  f"**Expected:** {item.expected}", "",
                  f"**Actual:** {item.actual or '(empty)'}", "",
                  f"**Missing:** {', '.join(item.missing_words) or 'None'}  ",
                  f"**Extra:** {', '.join(item.extra_words) or 'None'}  ",
                  f"**Substitutions:** {', '.join(item.substitutions) or 'None'}  ",
                  f"**Duplicated words:** {', '.join(item.duplicated_words) or 'None'}  ",
                  f"**Technical terms:** {', '.join(item.technical_term_problems) or 'None'}  ",
                  f"**Possible problem:** {item.possible_problem}"]
    if missing:
        lines += ["", "## Missing audio files", ""] + [f"- `{path}`" for path in missing]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="evaluation/cases.json")
    parser.add_argument("--report", default="evaluation/report.md")
    parser.add_argument("--json-report", default="evaluation/report.json")
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    cases = json.loads(manifest_path.read_text(encoding="utf-8"))["cases"]
    root = manifest_path.parent.parent
    missing = [case["audio"] for case in cases if not (root / case["audio"]).is_file()]
    runnable = [case for case in cases if case["audio"] not in missing]
    results = []
    if runnable:
        from asr_engine import WhisperModelProvider
        provider = WhisperModelProvider()
        for case in runnable:
            results.append(evaluate_case(case, root, provider))
    Path(args.report).write_text(render_markdown(results, missing), encoding="utf-8")
    Path(args.json_report).write_text(
        json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(render_markdown(results, missing))
    if missing:
        return 2
    return 1 if any(result.status == "FAIL" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
