# Data and generated runtime artifacts

- `test_audio/` contains tracked documentation and any intentionally permanent sample assets.
- `debug_audio/` is generated only when debug audio capture is enabled and is ignored.
- `runtime_logs/` contains application logs and is ignored.

The manifest-driven speech fixtures remain under `tests/speech_cases/` because they are
regression-test assets governed by `tests/expected/transcripts.json`.
