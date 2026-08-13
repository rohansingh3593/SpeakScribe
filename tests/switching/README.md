# Language-switch test suite

Run all switch levels and generate the timestamped Excel workbook:

```bash
pytest tests/switching/
```

Run a stage independently:

```bash
pytest tests/switching/light/
pytest tests/switching/complex/
pytest tests/switching/negative/
```

The suite exercises the production `RecognitionState`, `SpeechController`, queue,
`SpeechBufferWorker`, `ASRWorker`, and model-provider coordination. Deterministic ASR
doubles replace Whisper only where a concurrency/error outcome must be reproducible;
those cases do **not** claim acoustic accuracy or hardware latency. Microphone/Whisper
end-to-end accuracy remains in `tests/speech/` and requires generated audio, model
weights, and the host TTS/audio dependencies.

## Performance metric

`SW-PERF-001` and `SW-PERF-002` take 20 real `time.perf_counter()` measurements of
the transition orchestration path and preserve T0 through T4 for every run. The gate
is P95 switch-to-visible coordination latency <= 2.0 seconds. The workbook clearly
labels these as orchestration measurements rather than pretending that deterministic
text assignment is Whisper inference.

## Result semantics

* **PASS** — the application handled the scenario as required, including negative cases.
* **FAIL** — observed behavior violated the requirement.
* **XFAIL** — a documented production limitation was reproduced.
* **XPASS** — a strict expected failure unexpectedly passed and needs review.
* **ERROR** — the test could not complete.

The standard-library OOXML reporter generates
`test_reports/switching_test_report_<timestamp>.xlsx` with Test
Results, Performance Summary, Failure Summary, Summary, and raw Latency Runs sheets.
Failures link to the corresponding centralized `test_logs` session artifact.
