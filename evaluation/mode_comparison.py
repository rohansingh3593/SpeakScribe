"""Fair, measured FAST/BALANCED/ACCURATE comparison and report persistence."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean

from app.config.settings import PerformanceMode
from evaluation.evaluation_runner import EvaluationResult, evaluate_case
from evaluation.manifest_policy import load_manifest


MODES = tuple(PerformanceMode)
LOWER_IS_BETTER = {
    "wer", "first_partial_latency", "final_latency", "asr_time", "rtf",
    "cpu_percent", "memory_mb", "duplicate_partials", "dropped_chunks",
}
SCORE_WEIGHTS = {
    "accuracy": .40, "first_partial_latency": .20, "final_latency": .15,
    "technical_accuracy": .10, "language_accuracy": .10, "resource_usage": .05,
}


def _average(values, default=0.0):
    values = [value for value in values if value is not None]
    return mean(values) if values else default


def summarize(results: list[EvaluationResult]) -> dict:
    """Aggregate only measured values; missing partials remain explicitly null."""
    first = [item.first_partial_latency for item in results
             if item.first_partial_latency is not None]
    languages = {}
    for language in ("English", "Hindi", "Hinglish"):
        subset = [item for item in results if item.language == language]
        if subset:
            languages[language] = round(_average([item.similarity for item in subset]), 2)
    return {
        "test_cases": len(results),
        "accuracy": round(_average([item.similarity for item in results]), 2),
        "wer": round(_average([item.wer for item in results]), 4),
        "first_partial_latency": round(_average(first), 3) if first else None,
        "final_latency": round(_average([item.final_transcript_latency for item in results]), 3),
        "asr_time": round(_average([item.inference_seconds for item in results]), 3),
        "rtf": round(_average([item.real_time_factor for item in results]), 3),
        "cpu_percent": round(_average([item.cpu_percent for item in results]), 2),
        "memory_mb": round(_average([item.memory_mb for item in results]), 2),
        "technical_accuracy": round(_average(
            [item.technical_term_accuracy for item in results]), 2),
        "language_accuracy": round(_average(list(languages.values())), 2),
        "partial_updates": round(_average([item.partial_updates for item in results]), 2),
        "duplicate_partials": sum(item.duplicate_partials for item in results),
        "dropped_chunks": sum(item.dropped_chunks for item in results),
        "languages": languages,
    }


def _normalized(value, values, lower=False):
    low, high = min(values), max(values)
    if high == low:
        return 1.0
    score = (value - low) / (high - low)
    return 1.0 - score if lower else score


def add_scores(summaries: dict[str, dict]) -> None:
    def metric(mode, name, lower=False):
        available = [data[name] for data in summaries.values() if data[name] is not None]
        value = summaries[mode][name]
        return .5 if value is None or not available else _normalized(value, available, lower)

    for mode, data in summaries.items():
        resource = mean((metric(mode, "cpu_percent", True), metric(mode, "memory_mb", True)))
        score = (
            SCORE_WEIGHTS["accuracy"] * metric(mode, "accuracy") +
            SCORE_WEIGHTS["first_partial_latency"] * metric(mode, "first_partial_latency", True) +
            SCORE_WEIGHTS["final_latency"] * metric(mode, "final_latency", True) +
            SCORE_WEIGHTS["technical_accuracy"] * metric(mode, "technical_accuracy") +
            SCORE_WEIGHTS["language_accuracy"] * metric(mode, "language_accuracy") +
            SCORE_WEIGHTS["resource_usage"] * resource
        )
        data["suitability_score"] = round(score * 100, 1)


def build_comparison(by_mode: dict[str, list[EvaluationResult]]) -> dict:
    summaries = {mode: summarize(items) for mode, items in by_mode.items()}
    add_scores(summaries)
    best = {}
    for metric in ("accuracy", "wer", "first_partial_latency", "final_latency",
                   "asr_time", "rtf", "cpu_percent", "memory_mb",
                   "technical_accuracy", "language_accuracy", "suitability_score"):
        candidates = {mode: data[metric] for mode, data in summaries.items()
                      if data[metric] is not None}
        best[metric] = (min if metric in LOWER_IS_BETTER else max)(
            candidates, key=candidates.get) if candidates else None
    recommended = best["suitability_score"]
    language_recommendations = {}
    for language in ("English", "Hindi", "Hinglish"):
        candidates = {mode: data["languages"].get(language)
                      for mode, data in summaries.items()
                      if language in data["languages"]}
        if candidates:
            winner = max(candidates, key=candidates.get)
            language_recommendations[language] = {
                "mode": winner,
                "reason": (f"Highest measured {language} accuracy "
                           f"({candidates[winner]:.1f}%) in this run."),
            }
    transcripts = []
    case_ids = {item.case_id for items in by_mode.values() for item in items}
    for case_id in sorted(case_ids):
        rows = {mode: next((item for item in items if item.case_id == case_id), None)
                for mode, items in by_mode.items()}
        source = next((item for item in rows.values() if item), None)
        transcripts.append({"case_id": case_id, "audio": source.audio,
                            "language": source.language, "expected": source.expected,
                            "transcripts": {mode: item.actual if item else None
                                            for mode, item in rows.items()}})
    return {"generated_at": datetime.now(timezone.utc).isoformat(),
            "same_audio_for_all_modes": True, "weights": SCORE_WEIGHTS,
            "modes": summaries, "best_by_metric": best,
            "recommended_mode": recommended,
            "recommendation_reason": recommendation_reason(summaries, recommended),
            "language_recommendations": language_recommendations,
            "transcripts": transcripts}


def recommendation_reason(summaries, winner):
    data = summaries[winner]
    return (f"{winner.upper()} achieved the highest measured weighted score "
            f"({data['suitability_score']:.1f}/100), with {data['accuracy']:.1f}% accuracy "
            f"and {data['final_latency']:.3f}s mean final latency on this machine.")


def render_markdown(report: dict) -> str:
    modes = [mode.value for mode in MODES]
    labels = {
        "accuracy": "Accuracy", "wer": "WER", "first_partial_latency": "First partial (s)",
        "final_latency": "Final latency (s)", "asr_time": "ASR time (s)", "rtf": "RTF",
        "cpu_percent": "CPU (%)", "memory_mb": "RAM (MB)",
        "technical_accuracy": "Technical terms (%)",
        "language_accuracy": "Language accuracy (%)", "partial_updates": "Partial updates",
        "duplicate_partials": "Duplicate partials", "dropped_chunks": "Dropped chunks",
        "suitability_score": "Suitability (/100)",
    }
    lines = ["# Performance Mode Comparison", "", "All modes used the same audio files.", "",
             "| Metric | FAST | BALANCED | ACCURATE | Best |",
             "|---|---:|---:|---:|---:|"]
    for key, label in labels.items():
        values = [report["modes"][mode].get(key) for mode in modes]
        lines.append(f"| {label} | " + " | ".join("n/a" if value is None else str(value)
                                                       for value in values) +
                     f" | {(report['best_by_metric'].get(key) or 'n/a').upper()} |")
    lines += ["", f"**Recommended for this machine: {report['recommended_mode'].upper()}**", "",
              report["recommendation_reason"], "", "## Language recommendations", ""]
    for language, item in report["language_recommendations"].items():
        lines.append(f"- **{language}: {item['mode'].upper()}** — {item['reason']}")
    lines += ["", "## Side-by-side transcripts", ""]
    for case in report["transcripts"]:
        lines += [f"### {case['case_id']} ({case['language']})", "",
                  f"**Expected:** {case['expected']}", ""]
        for mode in modes:
            lines.append(f"- **{mode.upper()}:** {case['transcripts'][mode] or '(empty)'}")
        lines.append("")
    return "\n".join(lines)


def save_report(report: dict, output_root=Path("evaluation/mode_comparison")) -> Path:
    session = output_root / datetime.now().strftime("session_%Y%m%d_%H%M%S")
    session.mkdir(parents=True, exist_ok=False)
    (session / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    (session / "comparison.md").write_text(render_markdown(report), encoding="utf-8")
    with (session / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", *[mode.value for mode in MODES]])
        for metric in report["modes"]["fast"]:
            if metric not in {"languages"}:
                writer.writerow([metric, *[report["modes"][mode.value].get(metric)
                                           for mode in MODES]])
    return session


def run(cases: list[dict], root: Path, provider) -> dict:
    """Run every case under every mode; loop order guarantees a fair audio set."""
    by_mode = {mode.value: [] for mode in MODES}
    for case in cases:
        for mode in MODES:
            by_mode[mode.value].append(evaluate_case(case, root, provider, mode))
    return build_comparison(by_mode)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="tests/expected/transcripts.json")
    parser.add_argument("--output", default="evaluation/mode_comparison")
    args = parser.parse_args(argv)
    manifest = Path(args.manifest).resolve()
    _, cases = load_manifest(manifest)
    root = Path.cwd()
    missing = [case["audio"] for case in cases if not (root / case["audio"]).is_file()]
    if missing:
        parser.error(f"Missing {len(missing)} audio file(s); generate them first")
    from app.asr.asr_engine import WhisperModelProvider
    session = save_report(run(cases, root, WhisperModelProvider()), Path(args.output))
    print(f"Measured comparison saved to {session}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
