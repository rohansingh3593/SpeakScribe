"""Reusable prerecorded-audio accuracy and latency evaluation for SpeakScribe."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from difflib import SequenceMatcher
import json
import csv
import os
from pathlib import Path
import re
import sys
import time
import wave


TECHNICAL_TERMS = {
    "python", "pyqt6", "sqlalchemy", "alembic", "fastapi", "pydantic",
    "jenkins", "docker", "kubernetes", "git", "github", "gitlab", "jira",
    "api", "pr", "pytest", "postgresql", "mongodb", "kafka", "redis",
    "ctranslate2", "marianmt", "rest", "ci", "cd", "cpu", "gpu", "ram",
    "sql", "http", "https", "url", "json", "yaml", "aws", "faster", "whisper",
}


def format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


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
    audio_source: str
    scenario: str
    difficulty: str
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
    technical_term_accuracy: float
    number_accuracy: float
    punctuation_difference: bool
    total_processing_seconds: float
    first_partial_latency: float | None
    final_transcript_latency: float
    partial_updates: int
    duplicate_partials: int
    dropped_chunks: int
    root_cause: str
    recommended_fix: str
    possible_problem: str
    attempts: int
    initial_similarity: float
    retry_improvement: float
    best_retry_similarity: float | None
    best_retry_status: str | None
    quality_flags: list[str]


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
    if similarity >= 90:
        return "EXCELLENT"
    if similarity >= 80:
        return "PASS"
    if similarity >= 60:
        return "WARNING"
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


def _root_cause(case: dict, result: EvaluationResult) -> tuple[str, str]:
    features = set(case.get("features", []))
    if result.detected_language.casefold() != result.language.casefold() and result.language != "Hinglish":
        return "Language detection", "Review the language hint and detection window; do not rewrite the expected text."
    if result.technical_term_problems:
        return "Technical vocabulary", "Tune general vocabulary prompting and validate the whole technical category."
    if "low_volume" in features:
        return "Low-volume sensitivity", "Inspect RMS/VAD diagnostics and tune adaptive thresholds using all low-volume cases."
    if features.intersection({"background_noise", "traffic_noise", "office_noise", "fan_noise_high"}):
        return "Noise sensitivity", "Improve speech/noise discrimination and rerun the complete noise category."
    if features.intersection({"long_pause", "small_word_pauses", "silence_before", "silence_after"}):
        return "VAD", "Review pre-roll, hangover, and voiced-duration handling across the related pause cases."
    if result.duplicate_partials:
        return "Partial-result merging", "Improve stable-prefix merging without hardcoding this transcript."
    if result.real_time_factor > 1:
        return "Performance/latency", "Reduce redundant partial inference or select a generally faster decode profile."
    if result.punctuation_difference and result.similarity >= 80:
        return "Punctuation", "Adjust conservative final punctuation only; preserve recognized words."
    return "Whisper decoding", "Inspect segment confidence and audio quality, then rerun the scenario and full suite."


def evaluate_case(case: dict, root: Path, provider) -> EvaluationResult:
    from audio_pipeline import ASRJob
    from config import AppConfig, PerformanceMode

    path = root / case["audio"]
    # Forcing all code-switched speech through Hindi decoding drops English words.
    # Pure Hindi/English remain pinned; mixed Hinglish uses Whisper detection.
    language_hint = {"English": "en", "Hindi": "hi", "Hinglish": "auto"}[case["language"]]
    config = AppConfig(
        language_mode=language_hint,
        model_size=os.getenv("SPEAKSCRIBE_EVAL_MODEL", "small"),
        performance_mode=PerformanceMode.ACCURATE,
    )
    audio = load_wav(path, config.sample_rate)
    started = time.monotonic()
    partials: list[str] = []
    first_partial_latency = None
    if case.get("streaming", False):
        first_samples = min(len(audio), round(config.min_partial_duration * config.sample_rate))
        step_samples = max(1, round(max(config.partial_interval, 1.0) * config.sample_rate))
        for endpoint in range(first_samples, len(audio), step_samples):
            partial_job = ASRJob(audio=audio[:endpoint], final=False, utterance_id=1,
                                 captured_at=time.monotonic(),
                                 speech_seconds=endpoint / config.sample_rate)
            partial, _ = provider.get(config).transcribe(partial_job, "")
            if partial:
                partials.append(partial)
                if first_partial_latency is None:
                    first_partial_latency = time.monotonic() - started
    job = ASRJob(audio=audio, final=True, utterance_id=1, captured_at=time.monotonic(),
                 speech_seconds=len(audio) / config.sample_rate)
    final_started = time.monotonic()
    actual, detected = provider.get(config).transcribe(job, "")
    inference = time.monotonic() - final_started
    total_processing = time.monotonic() - started
    duration = len(audio) / config.sample_rate
    similarity, wer, edits = compare_transcripts(case["expected"], actual)
    expected_terms = TECHNICAL_TERMS.intersection(normalize_transcript(case["expected"]).split())
    actual_terms = set(normalize_transcript(actual).split())
    technical_problems = sorted(expected_terms - actual_terms)
    technical_accuracy = 100.0 if not expected_terms else 100 * (
        len(expected_terms) - len(technical_problems)) / len(expected_terms)
    expected_numbers = re.findall(r"\b\d+(?:[.:]\d+)*\b", normalize_transcript(case["expected"]))
    actual_numbers = re.findall(r"\b\d+(?:[.:]\d+)*\b", normalize_transcript(actual))
    number_accuracy = 100.0 if not expected_numbers else 100 * sum(
        (Counter(actual_numbers) & Counter(expected_numbers)).values()) / len(expected_numbers)
    expected_counts = Counter(normalize_transcript(case["expected"]).split())
    actual_counts = Counter(normalize_transcript(actual).split())
    duplicated = sorted(word for word, count in actual_counts.items()
                        if count > max(1, expected_counts[word]) + 1)
    result = EvaluationResult(
        case_id=case["id"], audio=case["audio"], language=case["language"],
        audio_source=case.get("audio_source", "human"),
        scenario=case.get("scenario", "unspecified"),
        difficulty=case.get("difficulty", "unspecified"),
        expected=case["expected"], actual=actual, detected_language=detected,
        similarity=similarity, wer=wer, status=status_for_similarity(similarity),
        duration_seconds=round(duration, 3), inference_seconds=round(inference, 3),
        real_time_factor=round(inference / max(duration, .001), 3),
        missing_words=edits.missing, extra_words=edits.extra,
        substitutions=edits.substitutions,
        duplicated_words=duplicated,
        technical_term_problems=technical_problems,
        technical_term_accuracy=round(technical_accuracy, 2),
        number_accuracy=round(number_accuracy, 2),
        punctuation_difference=(re.sub(r"[\w\s\u0900-\u097f]", "", case["expected"]) !=
                                re.sub(r"[\w\s\u0900-\u097f]", "", actual)),
        total_processing_seconds=round(total_processing, 3),
        first_partial_latency=(round(first_partial_latency, 3)
                               if first_partial_latency is not None else None),
        final_transcript_latency=round(inference, 3), partial_updates=len(partials),
        duplicate_partials=sum(a == b for a, b in zip(partials, partials[1:])),
        dropped_chunks=0, root_cause="", recommended_fix="", possible_problem="",
        attempts=1, initial_similarity=similarity, retry_improvement=0.0,
        best_retry_similarity=None, best_retry_status=None, quality_flags=[],
    )
    if not actual.strip():
        result.quality_flags.append("NO_TRANSCRIPTION")
    if detected.casefold() != case["language"].casefold() and case["language"] != "Hinglish":
        result.quality_flags.append("WRONG_LANGUAGE")
    if result.final_transcript_latency > 2.0:
        result.quality_flags.append("HIGH_LATENCY")
    if result.duplicate_partials:
        result.quality_flags.append("DUPLICATE_OUTPUT")
    if result.dropped_chunks:
        result.quality_flags.append("DROPPED_SPEECH")
    result.possible_problem = _diagnosis(result)
    result.root_cause, result.recommended_fix = _root_cause(case, result)
    return result


def evaluate_case_with_retries(case: dict, root: Path, provider, retries: int = 1,
                               evaluator=evaluate_case) -> EvaluationResult:
    """Rerun a failed case for evidence without hiding its initial failure."""
    primary = evaluator(case, root, provider)
    initial_similarity = primary.similarity
    best_retry = None
    attempts = 1
    while primary.status == "FAIL" and attempts <= max(0, retries):
        candidate = evaluator(case, root, provider)
        attempts += 1
        if best_retry is None or candidate.similarity > best_retry.similarity:
            best_retry = candidate
    primary.attempts = attempts
    primary.initial_similarity = initial_similarity
    if best_retry is not None:
        primary.best_retry_similarity = best_retry.similarity
        primary.best_retry_status = best_retry.status
        primary.retry_improvement = round(best_retry.similarity - initial_similarity, 2)
        if best_retry.status != primary.status:
            primary.quality_flags.append("UNSTABLE_RESULT")
    return primary


def render_markdown(results: list[EvaluationResult], missing: list[str],
                    baseline: dict[str, dict] | None = None) -> str:
    counts = defaultdict(int)
    by_language: dict[str, list[EvaluationResult]] = defaultdict(list)
    by_scenario: dict[str, list[EvaluationResult]] = defaultdict(list)
    by_difficulty: dict[str, list[EvaluationResult]] = defaultdict(list)
    by_source: dict[str, list[EvaluationResult]] = defaultdict(list)
    for result in results:
        counts[result.status] += 1
        by_language[result.language].append(result)
        by_scenario[result.scenario].append(result)
        by_difficulty[result.difficulty].append(result)
        by_source[result.audio_source].append(result)
    lines = ["# Speech Recognition Test Report", "",
             f"- Total configured: {len(results) + len(missing)}",
             f"- Executed: {len(results)}", f"- Missing audio: {len(missing)}",
             f"- Excellent: {counts['EXCELLENT']}", f"- Passed: {counts['PASS']}",
             f"- Warning: {counts['WARNING']}",
             f"- Failed: {counts['FAIL']}", ""]
    if results:
        accepted = counts["EXCELLENT"] + counts["PASS"] + counts["WARNING"]
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
    for source in ("human", "synthetic"):
        items = by_source[source]
        if items:
            lines.append(f"- {source.title()} audio average similarity: "
                         f"{sum(item.similarity for item in items) / len(items):.1f}%")
            lines.append(f"- {source.title()} audio average WER: "
                         f"{sum(item.wer for item in items) / len(items):.3f}")
    if results:
        lines += ["", "## Difficulty analysis", ""]
        for difficulty in ("easy", "medium", "hard", "extreme"):
            items = by_difficulty[difficulty]
            if items:
                lines.append(f"- {difficulty.title()}: "
                             f"{sum(item.similarity for item in items) / len(items):.1f}% "
                             f"(WER {sum(item.wer for item in items) / len(items):.3f})")
        lines += ["", "## Scenario analysis", ""]
        for scenario, items in sorted(by_scenario.items()):
            average = sum(item.similarity for item in items) / len(items)
            lines.append(f"### {scenario.replace('_', ' ').title()} — {average:.1f}%")
            lines.append("")
            for item in items:
                lines.append(f"- {item.case_id} ({item.language}): "
                             f"{item.similarity:.1f}% / WER {item.wer:.3f} / {item.status}")
            lines.append("")
        weakest = sorted(results, key=lambda item: item.similarity)[:10]
        slowest = sorted(results, key=lambda item: item.total_processing_seconds,
                         reverse=True)[:10]
        causes = Counter(item.root_cause for item in results
                         if item.status in {"WARNING", "FAIL"})
        technical = Counter(term for item in results for term in item.technical_term_problems)
        language_failures = [item for item in results
                             if item.detected_language.casefold() != item.language.casefold()]
        lines += ["## Top 10 weakest tests", ""] + [
            f"- {item.case_id}: {item.similarity:.1f}% ({item.status})" for item in weakest]
        lines += ["", "## Highest-latency tests", ""] + [
            f"- {item.case_id}: {item.total_processing_seconds:.2f}s total, "
            f"RTF {item.real_time_factor:.2f}" for item in slowest]
        lines += ["", "## Most common root causes", ""] + [
            f"- {cause}: {count}" for cause, count in causes.most_common()]
        lines += ["", "## Top technical-term errors", ""] + [
            f"- {term}: {count}" for term, count in technical.most_common(10)]
        lines += ["", "## Language-detection failures", ""] + [
            f"- {item.case_id}: expected {item.language}, detected {item.detected_language}"
            for item in language_failures]
        if baseline:
            comparisons = [(item, item.similarity - float(baseline[item.case_id]["similarity"]))
                           for item in results if item.case_id in baseline]
            lines += ["", "## Before vs after regression comparison", ""]
            for item, delta in sorted(comparisons, key=lambda pair: pair[1]):
                before = float(baseline[item.case_id]["similarity"])
                lines.append(f"- {item.case_id}: {before:.1f}% -> {item.similarity:.1f}% "
                             f"({delta:+.1f} points)")
    lines += ["", "| Test | Language | Source | Similarity | WER | Attempts | Best retry | Retry Δ | Flags | Inference | RTF | Status |",
              "|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|"]
    for item in results:
        lines.append(f"| {item.audio} | {item.language} | {item.audio_source.title()} | "
                     f"{item.similarity:.1f}% | "
                     f"{item.wer:.3f} | {item.attempts} | "
                     f"{item.best_retry_similarity if item.best_retry_similarity is not None else '-'} | "
                     f"{item.retry_improvement:+.1f} | {', '.join(item.quality_flags) or '-'} | "
                     f"{item.inference_seconds:.2f}s | "
                     f"{item.real_time_factor:.2f} | {item.status} |")
    for item in results:
        if item.status in {"EXCELLENT", "PASS"}:
            continue
        lines += ["", f"## {item.case_id}: {item.status}", "",
                  f"**Expected:** {item.expected}", "",
                  f"**Actual:** {item.actual or '(empty)'}", "",
                  f"**Missing:** {', '.join(item.missing_words) or 'None'}  ",
                  f"**Extra:** {', '.join(item.extra_words) or 'None'}  ",
                  f"**Substitutions:** {', '.join(item.substitutions) or 'None'}  ",
                  f"**Duplicated words:** {', '.join(item.duplicated_words) or 'None'}  ",
                  f"**Technical terms:** {', '.join(item.technical_term_problems) or 'None'}  ",
                  f"**Root cause:** {item.root_cause}  ",
                  f"**Recommended fix:** {item.recommended_fix}  ",
                  f"**Possible problem:** {item.possible_problem}"]
    if missing:
        lines += ["", "## Missing audio files", ""] + [f"- `{path}`" for path in missing]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="tests/expected/transcripts.json")
    parser.add_argument("--report", default="tests/results/latest_report.md")
    parser.add_argument("--json-report", default="tests/results/latest_report.json")
    parser.add_argument("--csv-report", default="tests/results/latest_report.csv")
    parser.add_argument("--baseline", help="Previous JSON report for regression comparison")
    parser.add_argument("--no-generate", action="store_true",
                        help="Do not synthesize missing WAV files before evaluation")
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    cases = json.loads(manifest_path.read_text(encoding="utf-8"))["cases"]
    root = Path.cwd()
    generation_errors = []
    if not args.no_generate:
        from tests.audio_generation import generate_all
        generation = generate_all(cases, root)
        sources = {record.audio_file: record.audio_source for record in generation}
        cases = [{**case, "audio_source": sources.get(case["audio"], "human")}
                 for case in cases]
        generation_errors = [record for record in generation
                             if record.status == "TTS_GENERATION_ERROR"]
    missing = [case["audio"] for case in cases if not (root / case["audio"]).is_file()]
    runnable = [case for case in cases if case["audio"] not in missing]
    baseline = None
    if args.baseline:
        previous = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        baseline = {item["case_id"]: item for item in previous}
    results = []
    if runnable:
        from asr_engine import WhisperModelProvider
        provider = WhisperModelProvider()
        evaluation_started = time.perf_counter()
        total = len(runnable)
        for index, case in enumerate(runnable, 1):
            case_started = time.perf_counter()
            retries = max(0, int(os.getenv("SPEAKSCRIBE_FAILED_RETRIES", "1")))
            results.append(evaluate_case_with_retries(case, root, provider, retries))
            elapsed = time.perf_counter() - evaluation_started
            eta = elapsed / index * (total - index)
            print(
                f"[ASR {index:03d}/{total:03d}] {case['id']} complete | "
                f"case={format_duration(time.perf_counter() - case_started)} "
                f"elapsed={format_duration(elapsed)} ETA={format_duration(eta)} "
                f"status={results[-1].status} attempts={results[-1].attempts}",
                file=sys.stderr, flush=True,
            )
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    report = render_markdown(results, missing, baseline)
    if generation_errors:
        report += "\n## TTS generation errors\n\n" + "\n".join(
            f"- TTS_GENERATION_ERROR {record.case_id}: {record.error}"
            for record in generation_errors) + "\n"
    Path(args.report).write_text(report, encoding="utf-8")
    Path(args.json_report).write_text(
        json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with Path(args.csv_report).open("w", newline="", encoding="utf-8-sig") as stream:
        fieldnames = list(asdict(results[0]).keys()) if results else [
            "case_id", "audio", "language", "scenario", "difficulty", "status"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            for key, value in row.items():
                if isinstance(value, list):
                    row[key] = " | ".join(value)
            writer.writerow(row)
    print(report)
    print(
        f"Evaluation completed at {datetime.now().astimezone().isoformat(timespec='seconds')} | "
        f"executed={len(results)}/{len(cases)}",
        file=sys.stderr, flush=True,
    )
    if generation_errors:
        return 3
    if missing:
        return 2
    return 1 if any(result.status == "FAIL" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
