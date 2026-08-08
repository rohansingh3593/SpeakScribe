# 120-case speech validation suite

`expected/transcripts.json` contains 30 scenarios with four genuinely different
variations each. The tracked `speech_cases/01_normal` through
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

Run the 120 mandatory parametrized pytest regressions:

```powershell
python -m pytest tests/test_transcription.py
```

There is no opt-in flag and no skip path. Pytest always collects all 120 cases. Every
missing WAV fails its corresponding case before Faster-Whisper is imported, so the
report shows exactly which recordings are absent and an incomplete corpus can never
look like a successful regression run.

Exit code 2 from the report runner means recordings are missing. That is an incomplete
dataset, not a successful recognition run. Real accuracy, latency, and fix/retest
claims must only be made from actual WAV inference.

For fix/retest regression comparison, preserve the previous JSON report and run:

```bash
python evaluation_runner.py --baseline tests/results/before_fix.json
```

The Markdown report lists every comparable before/after accuracy delta so an
improvement in one category cannot hide regressions elsewhere.
