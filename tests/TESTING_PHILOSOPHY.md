# Speech testing philosophy

These tests are an engineering diagnostic tool, not a demonstration. Expected text,
status thresholds, and difficult audio conditions must never be weakened to improve the
headline pass rate. Test-specific transcript substitutions are prohibited.

Every warning and failure remains in Markdown, JSON, and CSV reports. A diagnostic retry
does not upgrade the initial result. Reports separately expose no transcription, wrong
language, high latency, duplicate output, dropped speech, and unstable results.

## Required fix workflow

1. Preserve the current JSON report as the baseline.
2. Reproduce the failed case and inspect audio, VAD, queue, decoder, language, and text
   processing diagnostics rather than assuming Whisper is the cause.
3. Make a general implementation change, never an expected-text change.
4. Rerun the failed case and its four-case scenario.
5. Rerun English, Hindi, and Hinglish groups, then every configured case.
6. Compare accuracy, WER, first/final latency, memory, and regression flags.

An accuracy improvement accompanied by unacceptable latency, memory growth, crashes, or
another category regression is not an unconditional success. Failures are valuable
findings and must not be skipped, suppressed, or converted into ordinary passes.
