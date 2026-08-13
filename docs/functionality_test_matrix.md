# SpeakScribe functionality and test-protection matrix

**Baseline date:** 2026-08-13  
**Purpose:** connect every required behavior to its implementation, callers, current
tests, and gaps before extraction. This is an audit artifact, not a claim that
deterministic tests prove real acoustic quality or hardware latency.

## Matrix

| Capability | Behavioral-baseline implementation | Reusable-package implementation | Existing protection | Gap / required characterization before extraction |
|---|---|---|---|---|
| Microphone capture | `select_capture_device`, `AudioCaptureWorker.run` in `app/audio/audio_pipeline.py` | `capture_device`, `SoundCardRecorder` | `tests/integration/test_audio_pipeline.py`; `tests/unit/speakscribe/test_audio.py` | Real-device smoke evidence remains environment-specific. |
| System/loopback capture | Speaker-to-loopback selection in `AudioCaptureWorker`; desktop capture setting | `capture_device(..., "loopback")` | Capture-source integration test and library loopback tests | Installed-wheel loopback consumer smoke test is absent. |
| Downmix/normalization | Shared `speakscribe.audio.processor`; app statistics and ASR preparation | `to_mono`, `audio_normalization_gain`, `prepare_audio_for_asr` | Audio integration and package audio unit tests | Preserve the shared numerical path while moving app code. |
| Resampling/discontinuities | Capture batching, discontinuity recovery, `resample_audio_block` | Not present | Native-48-kHz and discontinuity integration tests | Must be extracted from app rather than replaced by the simpler recorder. |
| VAD | `EnergySpeechDetector.classify/reset` with adaptive noise floor/hysteresis | RMS threshold inside `StreamingBuffer` | Energy, quiet speech, adaptive voice/noise tests | Add direct long/fast/soft Hindi segmentation characterization using deterministic frames. |
| Speech buffering/pre-roll | `SpeechBufferWorker`, generation-namespaced utterances, pre-roll | `StreamingBuffer.push/flush` | Audio integration, switch first-frame, short-pause and public-service tests | Compare both implementations on identical frame sequences before consolidation. |
| Partial recognition | `SpeechBufferWorker._submit`, `ASRWorker`, `ComparisonASRWorker` | `SpeechToText.listen_continuously/start_continuous` | Mode comparison and service partial-before-finish tests | Add public event contract before GUI migration. |
| Final recognition | Final jobs, recovery decode and promotion in app ASR workers | Structured final `TranscriptionResult` | Mode tests, public `listen_once`, speech manifest | Public file transcription is absent. |
| FAST | Fast `DecodeProfile`, dedicated queue/lane and model sizing | No performance profile | Profile, queue, live-worker and Stop-running-fast tests | Add public config/API exposure and one-mode-failure isolation. |
| BALANCED | Balanced profile and refinement queue | No performance profile | Profile/refinement and Stop-running-balanced tests | Add public exposure; assert same immutable audio reaches all modes. |
| ACCURATE | Accurate profile, wider decode, refinement/evaluation | No performance profile | Profile/refinement, report and Stop-running-accurate tests | Add public exposure and explicit Accurate-failure isolation. |
| Progressive refinement | `ComparisonASRWorker`, `best_refinement_candidate`, segment-ID UI update | Not present | Fast/Balanced/Accurate refinement, empty/invalid/wrong-script promotion tests | Add UI-free event/order contract; do not expose Qt signal types. |
| English | Language settings, decoding policy, cleanup/vocabulary | `language_code`, `SpeechConfig.language` | 27 manifest cases; technical vocabulary; English hotword; Stop-to-English; Hindi-to-English switching | Add installed public API English test and real baseline report. |
| Hindi | Pinned language policy, Devanagari evidence, retry and transliteration protection | `language_code`, basic public language config | 30 manifest cases; prompt/hotword, Urdu/wrong-script retry, repetition cleanup, Hindi public-service case | Add public CPU-fallback and deterministic long/fast/soft/pause partial-to-final tests. |
| Hinglish/auto | Acoustic auto language, Hinglish detection, no global hotword bias | `None`/alias mapping to auto | 63 manifest cases; natural/multi-switch/technical cases; Hinglish service and text tests | Canonicalize accepted public names without changing current aliases. |
| Script handling | `script_metadata`, `apply_script_mode`, script validity and ITRANS lazy import | Not present | Extensive `test_text_processing.py`, mode and switching script tests | Extract intact; add structured script field to public results. |
| Translation | Lazy `_load` and queue-based `TranslationWorker` | Not present | No direct translation worker unit test | Characterize lazy loading, success, model error and Stop cleanup before extraction. |
| Text normalization | `clean_text`, technical canonicalization, quality filters | Only engine string join | Text-processing and evaluation normalization tests | Define raw vs normalized public fields before reuse. |
| Overlap/deduplication | `remove_history_overlap`, `incremental_transcript_delta`, live model | Not present | Overlap, revision, duplicate and repeated-word tests | Preserve deliberate repeated words while extracting. |
| Partial-to-final promotion | App ASR worker fallback and `LiveTranscriptModel.commit` | Service marks jobs final | Empty-final, valid-partial, wrong-script and live-model tests | Public events need same utterance identity and monotonic state. |
| Start | `SpeechController.start`, worker/capture setup | `SpeechToText.start` | Switching source/state and service tests | Add idempotent public start test and component-start failure cleanup. |
| Stop | `LiveSessionBoundary.stop`, non-blocking UI path | `SpeechToText.stop` with bounded joins | Full stopping suite and public cleanup tests | Semantics differ; retain UI responsiveness while creating deterministic library cleanup. |
| Restart | Controller retains model/capture coordination; generations | Public service recreates queues/events | Rapid Stop/Start, reusable generation, service session tests | Add explicit public same-instance restart test. |
| Language switching | `RecognitionState.switch`, controller queue/state reset | Not present | Light/complex/negative switching suites | Add public `set_language`/session method before GUI conversion. |
| Generation cancellation | `RecognitionState`, `LiveSessionBoundary`, worker `is_current` checks | Stop events only; no generation model | Stopping, stale partial/final and stress tests | Extract instance-owned generation token; avoid module global. |
| Stale-result protection | ASR workers and UI generation checks | No cross-session result token | Old-result, late-partial, stale-final and mixed-session tests | Required public event field and consumer-level test. |
| Queue coalescing/backpressure | Speech buffer and comparison mode queue replacement policies | Bounded audio/ASR queues | Integration queue tests, mode queue tests, queue-pressure switching tests | Known single-worker scheduling limitation remains xfailed. |
| Model reuse | `WhisperModelProvider` caches by model size/device | `FasterWhisperEngine._model` | Model reuse across switch/Stop and engine cleanup tests | Specify release semantics and multi-instance isolation. |
| CUDA-to-CPU fallback | `WhisperEngine._load_model`; package engine candidate loop | Same behavior in package engine | Indirect evaluation settings; no focused fake-backend fallback test | Add deterministic lazy CUDA failure → CPU/int8 test without model download. |
| Error mapping | Worker signals/logging and evaluation structured errors | `SpeakScribeError` hierarchy and chaining | Public capture/engine error tests; worker negative tests | Map mature model/device/config errors to stable public exceptions. |
| Standard logging | `app.utils.logger` structured sessions | Null-handler package logger and explicit configuration | Logger and suite-logging tests; lightweight-import test | Preserve quiet import while making rich diagnostics explicitly injectable. |
| Pipeline diagnostics | `PipelineDiagnostics`, root causes, debug audio, conservation | Not present | Full pipeline-diagnostics tests | Move behind an instance/explicit hook; eliminate `_ACTIVE` from public design. |
| Performance metrics | ASR/job timestamps, diagnostics, evaluation reports | Structured queue/inference/speech-to-result fields | Mode reports, service results, Stop/switch P95 tests | Save real before/after hardware baselines; deterministic orchestration is not inference. |
| Debug audio | `_write_debug_wav`, `PipelineDiagnostics.save_audio` | Not present | Background-writer and disabled-no-files tests | Keep opt-in and non-blocking; never enable on import. |
| PyQt integration | `SpeechSignals`, `MainWindow`, `FinalOnlyMainWindow` | External `examples/pyqt_recording_panel.py` consumes package | Template/source tests and package boundary test | Application controller still directly owns mature pipeline; needs event adapter. |
| Tkinter integration | Not part of app | External callback/generator example | Package boundary/location test | Add smoke/import test that does not require a display. |
| File transcription | Evaluation directly loads WAV and calls app engine | No public `transcribe_file` | Acoustic evaluation tests only | Required public API and external installed-wheel test are absent. |
| TTS/fixture provenance | `evaluation.audio_generation` | Out of runtime package | Audio-generation integration tests and manifest policy | Keep as development/test extra, never core runtime. |
| Evaluation/regression | Evaluation runner, comparison, manifest policy | Out of runtime package | Evaluation/mode/manifest/unit/integration tests | Preserve honest failure/retry/report semantics. |

## Coverage dimensions already present

- **Happy paths:** package audio/service tests, app unit tests, and manifest inference.
- **Edges:** quiet/loud/noisy audio, pauses, short/long utterances, script boundaries,
  empty results, repetitions, queue saturation, and generation rollover.
- **Negative/error paths:** missing devices/backends, worker exceptions, invalid text,
  TTS failure, stale results, shutdown switching, and manifest corruption.
- **Concurrency/state:** in-flight mode Stop, rapid switching, queue coalescing, capture
  continuation during ASR, generation invalidation, and model reuse.
- **Cleanup:** recorder/engine closing, Stop readiness, diagnostic writer closure,
  synthetic-audio cleanup, logger retention, and suite cleanup in `finally`.

## Known baseline limitations (must remain visible)

The focused baseline has two intentional xfails:

1. A single `ASRWorker` cannot allow a new Fast job to overtake already-running old
   inference.
2. `RecognitionState` currently accepts unknown language identifiers.

They are findings, not permission to delete or weaken tests. Fixes should be separate
behavior changes with explicit regression evidence.

## Next test-hardening phase (non-destructive)

Prioritize tests that close genuine public/core gaps rather than increasing count:

1. Public `SpeechRecognizer`/configuration/event contract with compatibility aliases.
2. Public file transcription using an injected deterministic engine plus WAV validation.
3. Same-instance Start → Stop → restart and language-switch generation semantics.
4. Deterministic CUDA initialization failure followed by CPU/int8 success.
5. Translation lazy-load/error/cleanup characterization.
6. Identical immutable audio delivered to all profiles, ordered refinement, and isolated
   mode failure.
7. Installed-wheel consumer executed outside the checkout with no repository on
   `PYTHONPATH`.
8. README/SPEAKSCRIBE code-block smoke tests.

Real Hindi, English, Hinglish, microphone, loopback, model and latency claims remain
release-environment gates in addition to these deterministic tests.
