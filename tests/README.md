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

TTS uses installed operating-system voices and never imports the ASR engine. Windows
uses `System.Speech` (install a `hi-IN` voice for Hindi; Hinglish prefers `hi-IN` then
`en-IN`), macOS uses `say` with Samantha/Lekha, and Linux requires `espeak-ng` or
`espeak`. A missing language-capable backend is reported as `TTS_GENERATION_ERROR`,
not as zero ASR accuracy. Generation metadata is stored in
`generated_audio_manifest.json`; an existing unmanaged WAV is treated as human and is
never overwritten, while `--regenerate` only recreates files previously marked
synthetic.

On Windows, inspect available voices with `python tests/generate_test_audio.py
--list-voices`. If `hi-IN` is absent, install **Hindi → Language options → Speech**
from Windows Settings, or run the elevated PowerShell command printed by the generator.
Latin-only Hinglish can fall back to an installed English voice; Devanagari never does.

Run the 120 mandatory parametrized pytest regressions:

```powershell
python -m pytest tests/test_transcription.py
```

There is no opt-in flag and no skip path. Pytest always collects all 120 cases. Every
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
