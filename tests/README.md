# Extensible speech validation suite

The binding diagnostic principles are documented in
[`TESTING_PHILOSOPHY.md`](TESTING_PHILOSOPHY.md). Tests intentionally seek breaking
points; expected transcripts and thresholds must not be changed to chase a pass rate.
The current grouped Hindi failure investigation and honest retest requirements are in
[`FAILURE_ANALYSIS.md`](FAILURE_ANALYSIS.md).

`expected/transcripts.json` starts with a 120-case baseline: 30 scenarios with four
genuinely different variations each. It is a minimum, not a fixed suite size. The tracked `speech_cases/01_normal` through
`speech_cases/30_combined` directories are the stable destinations for recordings.

Record the specified sentence and conditions for each case, preserving its filename.
Do not synthesize a passing transcript or change expected text after seeing ASR output.
Each case includes an `audio_profile` with target speaking rate, gain, noise type,
leading/trailing silence, and scenario-specific pause timing. Use clean source speech
plus reproducible non-speech overlays where appropriate; retain the clean master.

Run metadata and unit tests:

```bash
python -m pytest
```

Run all available production recordings and generate Markdown, JSON, and CSV reports:

```bash
python evaluation_runner.py
```

Missing files are synthesized automatically through the platform TTS backend. To
prepare audio only, or intentionally recreate files already marked synthetic, use:

```bash
python tests/generate_test_audio.py
python tests/generate_test_audio.py --regenerate
```

The one-command generation → validation → ASR → report workflow is:

```bash
python tests/run_speech_suite.py
```

The default console is a compact test-runner view: each case shows its language,
scenario, accuracy, WER, duration, moving-average ETA, and current PASS/WARNING/FAIL
totals. Technical audio, model, transcript, resource, and timing diagnostics are kept
in the timestamped file under `logs/` without flooding the terminal. Use `--debug` to
also show those diagnostics on the console, `--quiet` for errors plus the final summary,
or `--log-level LEVEL` for an explicit standard Python logging level. Every run writes
`logs/latest.log`, a timestamped complete log, and a warning/failure-only log beneath
`logs/errors/`.

ETA remains `calculating...` until three cases have completed, then uses the most recent
ten case durations so that an unusually fast or slow first case does not dominate it.
Tests over the configurable five-second default are marked `SLOW` but retain their
accuracy status. Ctrl+C saves the collected results as an `interrupted_report_*` set.

## Growing the suite

The first 120 entries are the preserved baseline, not a ceiling. Add a case when it
protects a distinct feature, interaction, boundary, production failure, or regression.
Every entry after the baseline must include `reason`, `feature`, and `type` metadata;
supported types are declared in the manifest growth policy. New cases are collected
automatically by both pytest and the evaluation runner, and progress totals are derived
from the manifest rather than hardcoded. Do not relax comparison thresholds, rewrite an
expected transcript, or add test-only recognition behavior to turn a useful failure green.

Recommended sequence to run every configured case, write reports, and then remove only
synthetic audio while retaining human recordings:

```powershell
python -m pip install -r requirements.txt
python -c "from faster_whisper import WhisperModel; print('faster-whisper ready')"
python -m pytest -q tests/test_audio_generation.py tests/test_validation_manifest.py tests/test_evaluation.py tests/test_text_processing.py
python tests/run_speech_suite.py --cleanup-after generated
```

Always use `python -m pip`, not a bare `pip`, so packages are installed into the same
virtual environment that runs the suite. A missing ASR dependency now stops once with
`ASR_DEPENDENCY_ERROR` before evaluation, instead of producing one identical failure per case.

The command prints start/completion timestamps, dynamic `[001/NNN]` progress,
elapsed time, a continuously updated ETA, per-stage duration, final suite status, total
execution time, and report locations. The ETA becomes meaningful after several cases;
runtime depends on audio length and hardware, and CPU inference can take substantially
longer than CUDA inference.

The evaluator defaults to the multilingual `small` model and accurate decoding. Set
`SPEAKSCRIBE_EVAL_MODEL=medium` for a slower accuracy-focused run or `base` for a faster
diagnostic run. Generated WAV metadata is versioned; audio made by an older generator
is rebuilt automatically so stale English-voice Hindi files are not silently reused.

Every case below 60% is automatically run one additional time for diagnostic evidence.
The initial failure remains the official result even if a retry passes; the report marks
that case `UNSTABLE_RESULT` and records attempts, initial similarity, best retry, and
retry improvement. Set `$env:SPEAKSCRIBE_FAILED_RETRIES = "2"` to allow two retries,
or `"0"` to disable them. Retries never substitute expected text or change test data.

Reports preserve explicit `NO_TRANSCRIPTION`, `WRONG_LANGUAGE`, `HIGH_LATENCY`,
`DUPLICATE_OUTPUT`, `DROPPED_SPEECH`, and `UNSTABLE_RESULT` quality flags. Accuracy
status is never upgraded merely because a retry happened to produce a better result.

The cleanup runs in a `finally` block, including when generation or evaluation fails.
To explicitly delete human recordings too, use `--cleanup-after all`. Without a cleanup
option, audio is retained for the next run. Do not run `test_transcription.py` after
`run_speech_suite.py` unless a second complete ASR run is intended.

TTS is independent from recognition and never imports the ASR engine itself. Windows
uses neural Hindi synthesis with a `System.Speech` fallback, macOS uses `say` with
Samantha/Lekha, and Linux requires `espeak-ng` or
`espeak`. A missing language-capable backend is reported as `TTS_GENERATION_ERROR`,
not as zero ASR accuracy. Generation metadata is stored in
`generated_audio_manifest.json`; an existing unmanaged WAV is treated as human and is
never overwritten, while `--regenerate` only recreates files previously marked
synthetic.

On Windows, inspect available voices with `python tests/generate_test_audio.py
--list-voices`. If `hi-IN` is absent, install **Hindi → Language options → Speech**
from Windows Settings, or run the elevated PowerShell command printed by the generator.
Latin-only Hinglish can fall back to an installed English voice; Devanagari never does.
On Windows, Hindi and Devanagari fixtures use Microsoft neural `edge-tts` first because
the legacy SAPI Hindi voice produced systematic pronunciation artifacts that obscured
ASR regressions. An installed `hi-IN` SAPI voice remains the offline fallback. English
continues to prefer an installed SAPI voice. Install updated dependencies with
`python -m pip install -r requirements.txt`; neural synthesis needs internet access only
while creating missing or stale synthetic WAV files. Generator-version changes rebuild
managed synthetic audio but never overwrite human recordings.

Remove only synthetic/generated test audio (human recordings are preserved):

```powershell
python tests/generate_test_audio.py --remove-generated
```

To intentionally remove all manifest audio files, including any human recordings:

```powershell
python tests/generate_test_audio.py --remove-all
```

Run every mandatory, manifest-driven parametrized ASR regression:

```powershell
python -m pytest tests/test_transcription.py
```

There is no opt-in flag and no skip path. Pytest always collects every manifest case. Every
missing WAV is generated and validated before Faster-Whisper is imported. If the TTS
backend or a compatible language voice is unavailable, the case fails explicitly as
`TTS_GENERATION_ERROR` rather than being skipped or counted as an ASR accuracy failure.

Exit code 2 means generation was disabled and recordings remain missing; exit code 3
means TTS generation failed. Neither is a successful recognition run. Real accuracy,
latency, and fix/retest claims must only be made from actual WAV inference.

For fix/retest regression comparison, preserve the previous JSON report and run:

```bash
python evaluation_runner.py --baseline tests/results/before_fix.json
```

The Markdown report lists every comparable before/after accuracy delta so an
improvement in one category cannot hide regressions elsewhere.
