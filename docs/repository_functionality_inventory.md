# SpeakScribe repository audit and functionality inventory

**Audit date:** 2026-08-13  
**Audit scope:** the complete tracked repository, including application and library
Python sources, launchers, examples, evaluation utilities, test infrastructure,
metadata, manifests, and documentation.  
**Change boundary:** this document is the Phase 1 audit deliverable. No production
code, package layout, dependency metadata, or tests were changed during the audit.

## Executive findings

The repository is not an un-packaged application awaiting its first package skeleton.
It already contains two related implementations:

1. The mature behavioral baseline under `app/`, including the PyQt application,
   adaptive VAD, three-profile scheduling, language/session generation boundaries,
   script processing, translation, and extensive diagnostics.
2. A smaller, already installable `src/speakscribe` package at version `0.1.0`, with
   a GUI-free public API, lazy optional backends, structured results, live buffering,
   examples, and focused unit tests.

The desktop application has **not** yet become a consumer of the complete public
library. It imports only shared audio preprocessing from `speakscribe`; most production
behavior remains independently implemented under `app`. A safe next phase should
therefore converge the two implementations behind adapters and characterization tests,
not move files wholesale or replace the mature application pipeline with the smaller
library pipeline.

The initial worktree was not clean: `src/speakscribe.egg-info/` was present as an
untracked generated directory before this audit. It was neither edited nor removed.
This prevents claiming a pristine Phase 0 baseline until its owner decides whether it
should be ignored or deleted.

## 1. Repository tree

```text
SpeakScribe/
├── app/                              # Mature desktop/application implementation
│   ├── asr/
│   │   ├── asr_engine.py             # Whisper lifecycle and ASR workers
│   │   ├── language_transition.py    # Language/generation state
│   │   └── session_transition.py     # Immediate Stop boundary
│   ├── audio/audio_pipeline.py       # Capture, adaptive VAD, buffering, queues
│   ├── config/
│   │   ├── decoding_policy.py        # Prompt/hotword/recovery policy
│   │   ├── settings.py               # Runtime and performance profiles
│   │   └── technical_vocabulary.py   # Vocabulary and topic detection
│   ├── processing/
│   │   ├── live_transcript.py        # Partial/final transcript state
│   │   ├── technical_questions.py    # Sentence/question extraction
│   │   ├── text_processing.py        # Script, cleanup, overlap, refinement
│   │   └── translation.py            # Lazy Marian translation worker
│   ├── utils/
│   │   ├── logger.py                 # Structured application logging
│   │   └── pipeline_diagnostics.py   # Stage traces and debug audio
│   ├── main.py                       # Controller and full PyQt UI
│   └── final_only_main.py            # Default final-only UI variant
├── src/speakscribe/                  # Existing reusable package
│   ├── audio/                        # Device selection, processing, recorder
│   ├── logging/                      # Opt-in standard logging configuration
│   ├── services/speech_service.py    # Instance-based public orchestration
│   ├── transcription/                # Engine interface, Whisper, streaming buffer
│   ├── config.py                     # Frozen typed public configuration
│   ├── exceptions.py                 # Domain exceptions
│   ├── models.py                     # Structured result model
│   └── __init__.py                   # Public exports
├── evaluation/                       # Generation, scoring, comparison, reports
├── examples/                         # CLI, Tkinter, PyQt, and app templates
├── tests/
│   ├── expected/transcripts.json     # 120-case behavioral manifest
│   ├── fixtures/                     # Metrics and observability infrastructure
│   ├── integration/                  # Audio pipeline/generation/runner integration
│   ├── speech/                       # Mandatory manifest-driven acoustic tests
│   ├── stopping/                     # Stop/session cancellation regressions
│   ├── switching/                    # Light, complex, negative, stress, latency
│   └── unit/                         # App, library, evaluation, logging, text tests
├── data/                             # Audio location policy; WAVs intentionally absent
├── scripts/                          # Operational-script policy placeholder
├── main.py                           # Compatibility launcher
├── evaluation_runner.py              # Compatibility facade/launcher
├── pyproject.toml                    # Existing setuptools package metadata, 0.1.0
├── pytest.ini                        # Root and src import paths
├── requirements.txt                  # Full application/development environment
├── README.md                         # Application and library overview
└── SPEAKSCRIBE.md                    # Existing package/API guide
```

No tracked PowerShell (`.ps1`) files are present. PowerShell commands are documented
in `tests/README.md`; operational scripts otherwise have only `scripts/README.md`.
The repository contains no `setup.py` or `setup.cfg`; `pyproject.toml` is the packaging
source. There is no dedicated debug executable: debug behavior is provided by CLI
flags, structured loggers, pipeline diagnostics, and evaluation/test artifacts.

## 2. Current repository analysis by production file

### Application runtime

| File | Responsibilities and public behavior | Used by | Change risk |
|---|---|---|---|
| `main.py` | Compatibility launcher for the final-only desktop app. | Users/scripts. | Breaking the historical command. |
| `app/final_only_main.py` | Subclasses the full window to hide processing and ignore partial display; creates `QApplication`. | Root launcher and launcher-template tests. | Default UI behavior changes. |
| `app/main.py` | `SpeechSignals`, `SpeechController`, full `MainWindow`, start/stop/restart, language switching, queue/workers, translation dispatch, transcript promotion, performance comparison, CLI/debug startup. | Both PyQt templates and many source-characterization tests. | Highest coupling risk: orchestration and presentation coexist; Stop is intentionally non-blocking. |
| `app/audio/audio_pipeline.py` | Capture-device selection, loopback/microphone capture, 48 kHz to 16 kHz resampling, statistics, adaptive energy detector, warm-up, pre-roll, speech segmentation, rolling partials, finals, generation-aware utterance IDs, bounded queue policy. Reuses library audio preparation. | App controller, ASR engine types, integration/switching/stopping tests. | Audio loss, latency, queue starvation, incorrect device, or VAD regression. |
| `app/asr/asr_engine.py` | Debug WAVs, CUDA-to-CPU model loading, Whisper decoding/recovery, prompt/hotword policy, cleanup/script validation, model reuse, single and three-profile workers, stale generation checks, partial/final promotion and diagnostics. | App, evaluation, stopping, switching, mode tests. | Recognition quality, script correctness, stale output, CPU latency, model reloads. |
| `app/asr/language_transition.py` | Lock-protected `RecognitionState` snapshots, monotonic generations, language/script switching and current-generation checks. | Controller, workers, Stop/switch suites. | Cross-session data leakage and stale text. |
| `app/asr/session_transition.py` | `LiveSessionBoundary`, immediate capture disable, generation invalidation, queue clearing, measured Stop readiness. | Controller and Stop tests. | UI hangs or old work becoming visible. |
| `app/config/settings.py` | `PerformanceMode`, distinct Fast/Balanced/Accurate decode profiles, complete mutable `AppConfig`, audio/VAD/queue/model/script/translation defaults, validation. | Nearly all application stages and tests. | Default behavior and latency/quality balance. |
| `app/config/decoding_policy.py` | Context-only initial prompts, language-safe hotwords, relaxed retry thresholds. | ASR engine and focused regression tests. | Hindi script corruption, Hinglish bias, hallucination. |
| `app/config/technical_vocabulary.py` | Large domain vocabulary and word-boundary topic detection. | Settings and question processing. | Technical term fidelity and categorization. |
| `app/processing/text_processing.py` | Script metrics, Hindi/English/Hinglish detection, canonical technical spelling, conservative cleanup, quality rejection, overlap/deduplication, incremental deltas, profile agreement/diffs, refinement selection, optional lazy ITRANS conversion. | ASR, UI, evaluation, extensive tests. | Destructive transcript mutation and Hindi/Hinglish regression. |
| `app/processing/live_transcript.py` | UI-independent in-memory partial/final/refinement model and paragraph policy. | UI-focused unit tests; conceptually reusable. | Duplicate/missing finals and history mutation. |
| `app/processing/technical_questions.py` | Rule-based sentence splitting and technical question classification; optional caller-supplied semantic classifier. | Unit tests and future consumers. | False question classification. |
| `app/processing/translation.py` | Background queue worker; lazy Marian Hindi-to-English model/tokenizer load. | `SpeechController`. | Heavy dependency/loading, shutdown, translation errors. |
| `app/utils/logger.py` | Standard logging hierarchy, session/module/repository logs, retention, status events/generators and compatibility helpers. | App, evaluation, diagnostics, logger tests. | Forced I/O, duplicate handlers, lost diagnostics. |
| `app/utils/pipeline_diagnostics.py` | Non-blocking lifecycle events, root-cause classification, conservation of detected utterances, optional async debug WAV output and session summaries. | Audio/ASR/UI and diagnostics tests. | Missing evidence, blocking real-time work, dropped trace attribution. |

### Existing reusable package

| File | Responsibilities and public behavior | Used by | Change risk |
|---|---|---|---|
| `src/speakscribe/__init__.py` | Public exports: `SpeechToText`, `SpeechConfig`, `TranscriptionResult`, engine base/concrete classes and domain errors. | External-style examples and package tests. | Public API compatibility and import weight. |
| `src/speakscribe/config.py` | Frozen `SpeechConfig` with capture, streaming, queues, model/device and beam settings plus validation. | Public service/audio/engine/tests. | Public constructor compatibility; currently lacks app performance/script/translation fields. |
| `src/speakscribe/models.py` | Frozen `TranscriptionResult` with text/language/confidence/finality, timestamp, utterance and latency data. | Public service/engine/tests. | Consumer data contract. |
| `src/speakscribe/exceptions.py` | Base, microphone/device, transcription and service-state errors; legacy alias. | All package layers and public exports. | Error handling compatibility. |
| `src/speakscribe/audio/processor.py` | Mono conversion, RMS, bounded normalization/preparation; alias retained. | Package and mature app pipeline. | Shared numerical behavior affects both implementations. |
| `src/speakscribe/audio/microphone.py` | Lazy SoundCard lookup and microphone/loopback selection with domain errors. | Recorder/audio tests. | Platform device semantics. |
| `src/speakscribe/audio/recorder.py` | Abstract recorder and instance-based `SoundCardRecorder`; start/iterate/stop/close. | Public service and injected test doubles. | Capture cleanup and system-audio mapping. |
| `src/speakscribe/transcription/base.py` | Injectable transcription engine protocol. | Public API/service/tests. | Third-party backend compatibility. |
| `src/speakscribe/transcription/engine.py` | Lazy Faster-Whisper model; auto CUDA/float16 then CPU/int8 fallback; structured result. | Public service. | Model lifecycle and fallback behavior. |
| `src/speakscribe/transcription/streaming.py` | RMS-based pre-roll, partial/final jobs, silence and maximum-utterance finalization. | Public service tests. | Simpler than mature adaptive app VAD; swapping it in would regress behavior. |
| `src/speakscribe/services/speech_service.py` | Instance-owned queues/events/workers, generator and callback APIs, injected recorder/engine, deterministic lifecycle/context manager. | Examples and public API tests. | Concurrency, job priority, deadlocks, cleanup. |
| `src/speakscribe/logging/logger.py` | Null-handler logger by default and explicit `basicConfig`. | Package layers. | Import must remain side-effect-light. |
| `src/speakscribe/utils/helpers.py` | Maps public language aliases to Whisper codes. | Package Whisper engine. | Language semantics. |

### Evaluation, test, and examples

| Area | Responsibilities | Risks |
|---|---|---|
| `evaluation/audio_generation.py` | Cross-platform TTS, neural Hindi preference, audio profiles, generated/human provenance and safe cleanup. | Invalid fixtures can masquerade as ASR failures. |
| `evaluation/evaluation_runner.py` | WAV loading, real inference, transcript normalization/alignment/WER, language/script evidence, retries that never hide initial failure, regression reports. | Misleading quality claims or hidden regressions. |
| `evaluation/mode_comparison.py` | Runs all three modes, aggregates resource/latency/accuracy metrics, weighted recommendation and JSON/CSV/Markdown reports. | Fabricated or incomparable profile claims. |
| `evaluation/manifest_policy.py` | Protects the 120-case baseline and requires metadata for extensions. | Silent test deletion or weakened coverage. |
| `evaluation/suite_logging.py` | Progress/ETA, result aggregation and report log locations. | Loss of diagnostic visibility. |
| `tests/fixtures/pytest_observability.py` and `tests/conftest.py` | One session per pytest run; lifecycle, captured output, expected/actual/metrics and failure artifacts. | Central testing philosophy would be weakened by bypassing hooks. |
| `examples/cli_example.py`, `simple_example.py`, `tkinter_example.py`, `pyqt_recording_panel.py` | External consumers of `speakscribe`; callback/generator GUI integration. | Important evidence for UI independence. |
| `examples/live_update_main.py`, `performance_comparison_template.py` | Consumers of mature `app` implementation. | Preserve application examples during migration. |

## 3. Complete functionality inventory

### Audio and speech segmentation

- Physical microphone and system/loopback capture through SoundCard.
- Default-speaker-to-loopback mapping and microphone selection.
- Multichannel downmix, DC centering, safe low-level gain, invalid-sample metrics.
- Native capture batching, discontinuity recovery, warm-up, and 48 kHz-to-16 kHz
  resampling in the mature app pipeline.
- Adaptive RMS/noise-floor hysteresis, speech start/continuation/release thresholds,
  pre-roll, minimum speech, silence finalization, rolling maximum duration.
- Frequent live partial jobs and stable final jobs.
- Immutable timestamped/generation-aware `ASRJob` data and bounded queue policies.
- A separate simpler public `StreamingBuffer`; this overlap must be characterized
  before consolidation.

### Recognition and performance

- Lazy Faster-Whisper loading and explicit model reuse.
- Automatic CUDA/float16 attempt with CPU/int8 fallback.
- Fast, Balanced, and Accurate modes with distinct windows, partial cadence, context,
  beams, temperatures and decoding thresholds.
- Fast live lane plus Balanced/Accurate refinement, or explicit three-mode comparison.
- Queue-wait, inference, speech-to-result, first-text, final, CPU and memory metrics.
- Freshness logging/handling and bounded backlog policies.
- Recovery decode for empty/invalid final candidates using relaxed rejection gates.

### Languages, scripts, translation, and text

- Pinned English (`en`), pinned Hindi (`hi`), automatic/Hinglish code switching, and
  equivalent display labels.
- Original, Latin and Devanagari script modes; Devanagari/Latin/Arabic script evidence.
- Hindi safety rules that avoid English prompts/Latin hotwords and reject Urdu-script
  corruption; Hinglish avoids unconditional hotword bias.
- Optional lazy ITRANS transliteration that protects technical vocabulary.
- Optional background Hindi-to-English Marian translation.
- Technical vocabulary canonicalization and topic/question detection.
- Conservative cleanup, low-quality/hallucination rejection, overlap removal,
  deduplication, suffix deltas, partial replacement, final promotion, paragraphing,
  and cross-profile agreement/diff display.

### Runtime and lifecycle

- Start, Stop, restart, continuous generator, callbacks and context-manager cleanup in
  the package.
- Desktop Stop immediately disables capture, invalidates the generation and clears
  queued work without waiting for in-flight CTranslate2 inference on the UI thread.
- Language/script switching uses lock-protected monotonic generations.
- Stale partial/final suppression, session-namespaced utterance IDs and preservation of
  finalized history across Stop and switches.
- Instance-owned state in `SpeechToText`; mostly controller/worker-owned state in the
  application, with limited logger/diagnostics globals documented below.

### UI, logging, diagnostics, and evaluation

- PyQt final-only default UI and retained processing/live-refinement UI template.
- Compact record controls, language shortcuts, settings strip, timer, status/level
  indicators, selectable/copyable transcript, segment table and comparison view.
- Standalone PyQt and Tkinter examples using the public package.
- Standard Python logger names, opt-in lightweight package logging, rich application
  log sessions, per-module/repository logs, retention and status streams.
- Optional pipeline stage traces, background debug-audio WAVs, root-cause summaries,
  queue/VAD/ASR/finalization diagnostics.
- 120-case English/Hindi/Hinglish acoustic manifest over 30 scenarios and four
  variations, TTS generation/provenance, WER/similarity, latency/resources, retries,
  mode comparison and before/after regression analysis.

## 4. Existing tests inventory and baseline

`python -m pytest --collect-only -q` collected **363 tests**. The manifest contributes
**120 mandatory speech tests**: 27 English, 30 Hindi and 63 Hinglish cases, covering 30
scenarios with four distinct variations each.

| Suite | Coverage |
|---|---|
| `tests/unit/speakscribe/` | Lightweight imports, no GUI in core, audio/device behavior, structured results, live partial/final service, callbacks, queue pressure, cleanup, validation and error chaining. |
| Other `tests/unit/` | Three modes, decoding policy, text/script/dedup, live UI model, logging, diagnostics, evaluation, manifest preservation, package structure, launchers and observability. |
| `tests/integration/` | Capture/VAD/resampling/device policy, audio preparation, queue policy, TTS provenance/backends, suite command and cleanup behavior. |
| `tests/stopping/` | Immediate Stop, generation invalidation, queue clearing, model reuse, rapid cycles, language after Stop, P95 gate and no UI join/wait. |
| `tests/switching/` | Basic transitions, P95 coordination, state/model/capture reuse, script pairing, concurrency, post-switch audio, failure and stress scenarios, workbook reporting. |
| `tests/speech/` | Real generated/recorded WAV inference; all manifest cases are collected with no skip path. |

No existing test was deleted, skipped, marked xfail, or altered in this audit. A full
acoustic baseline was not run during the documentation-only pass because tracked WAV
fixtures are absent and running that suite may synthesize 120 files and download/load
large models. Before structural work, run the phase-gate commands below in the intended
release hardware/environment and preserve its JSON report. Collection itself passed.

## 5. Current dependency graph

```text
pyproject core: numpy
├── audio extra: soundcard
├── whisper extra: faster-whisper
├── pyqt extra: PyQt6
├── all extra: soundcard + faster-whisper
└── dev extra: pytest

full requirements.txt
├── editable package
├── runtime: numpy, soundcard, psutil, faster-whisper
├── GUI: PyQt6
├── script/translation: indic-transliteration, transformers, torch, sentencepiece
└── test fixture generation: edge-tts
```

Internal dependency direction currently is:

```text
src/speakscribe config + models + exceptions
    -> audio processing/device/recorder
    -> transcription base/streaming/Whisper
    -> SpeechToText public service

app config + diagnostics + shared speakscribe.audio.processor
    -> app audio and text/language policy
    -> app ASR/session workers
    -> SpeechController (inside app.main)
    -> PyQt MainWindow/final-only variant

evaluation -> app config/text/ASR behavior
examples -> either speakscribe public API or outer app UI
```

Important metadata gaps: the `all` extra excludes PyQt, translation, transliteration,
psutil and fixture-generation dependencies; no `translation` extra exists; and
`__version__` is not exported even though `pyproject.toml` already declares `0.1.0`.
These are recommendations for review, not changes made by this audit.

## 6. UI-specific modules

- `app/main.py`: `SpeechSignals` and `MainWindow` are PyQt-specific; it also contains
  the non-visual `SpeechController`, which should eventually move behind or adapt to the
  library API.
- `app/final_only_main.py`: PyQt application/window variant.
- `examples/pyqt_recording_panel.py`: external PyQt consumer of the public API.
- `examples/performance_comparison_template.py`: PyQt consumer of `app.MainWindow`.
- `examples/live_update_main.py`: launcher for the PyQt application implementation.
- `main.py`: indirect PyQt launcher.
- `examples/tkinter_example.py`: non-PyQt GUI consumer, correctly outside the core.

No module under `src/speakscribe` imports PyQt or Tkinter.

## 7. Functions and classes coupled to PyQt

- Directly coupled: `SpeechSignals`, all `MainWindow` methods, `FinalOnlyMainWindow`,
  both UI `main()` functions, and all `RecordingPanel` methods.
- Structurally coupled by location/signals: `SpeechController` is a plain Python class,
  but accepts and emits through a PyQt `SpeechSignals` object and is defined in the UI
  module. Its `preload_model`, `start`, `switch_recognition_language`, `start_stream`,
  `translate`, `reset_live_session`, and `stop` methods need a callback/event adapter
  before extraction.
- Application workers accept a signal bundle and call `.emit`; their speech logic is
  not inherently Qt-specific, but the current interface makes `ASRWorker`,
  `ComparisonASRWorker`, and controller wiring indirectly coupled to Qt consumers.
- `LiveTranscriptModel`, text processing, audio pipeline, recognition/session state,
  diagnostics and configuration contain no PyQt imports and are core candidates.

## 8. Global/shared runtime state

### Mutable process-wide state

- `app.utils.logger._SESSION`: active structured application logging session.
- `app.utils.pipeline_diagnostics._ACTIVE`: active diagnostic collector, replaced by
  `configure_pipeline_diagnostics()`.
- `app.main.DEBUG_DIAGNOSTICS`: CLI-set process-wide debug flag.
- Python logging registry/handlers in both logging implementations.

### Instance-owned state that should remain instance-owned

- `SpeechToText`: recorder, engine, events, queues, locks, workers, callback worker and
  running state.
- `SpeechController`: configuration, queues, state, model provider, worker and stop
  event references.
- `RecognitionState`: lock, generation, language, script and active flag.
- `WhisperModelProvider`: lock and cached model(s).
- Audio/ASR workers, `StreamingBuffer`, `LiveTranscriptModel`, `TranslationWorker`, and
  `PipelineDiagnostics` own their mutable state.

The remaining module constants are immutable enums, tuples/frozensets, regexes,
threshold/profile dictionaries or logger adapters. Profile/vocabulary dictionaries are
technically mutable, so public callers should not be encouraged to mutate them.

## 9. Proposed small public API

Preserve the already-shipped names while adding clearer aliases and the missing mature
capabilities incrementally:

```python
from speakscribe import (
    SpeechRecognizer,       # preferred alias/new facade
    SpeechToText,           # retained 0.x compatibility name
    RecognizerConfig,       # preferred alias/evolution of SpeechConfig
    SpeechConfig,           # retained compatibility name
    TranscriptionEvent,
    TranscriptionResult,
    PerformanceMode,
    SpeakScribeError,
)

config = RecognizerConfig(
    language="hi",                 # en, hi, hinglish/auto
    performance="balanced",        # fast, balanced, accurate
    script="devanagari",           # original, devanagari, latin
    capture_source="microphone",    # microphone, loopback
    device="auto",
    compute_type="auto",
    debug=False,
)

with SpeechRecognizer(config) as recognizer:
    for event in recognizer.listen():
        if event.is_partial:
            handle_partial(event)
        else:
            handle_final(event)
```

Primary live style should remain a synchronous iterator because the existing
`listen_continuously()` API and examples already establish it; callbacks remain a
convenience adapter for GUI toolkits. A future `transcribe_file(path)` should reuse the
same engine/policy and structured result rather than create a third decode path.

Recommended stable model fields:

- `TranscriptionResult`: current fields plus requested/effective language, script,
  performance mode, start/end time and optional generation/session/diagnostic IDs.
- `TranscriptionEvent`: utterance ID, generation, text, state (`partial`, `final`,
  `refinement`, `stale`, `error`), mode and timings; `is_partial`/`is_final` properties.
- Preserve raw recognition text separately if normalized/display text is introduced.

Do not publicly expose internal queues, Qt signals, `ASRJob`, worker classes, Whisper
model objects or application widgets.

## 10. Proposed package structure

Adapt the existing package instead of replacing it:

```text
src/speakscribe/
├── __init__.py                 # intentionally small stable exports + __version__
├── api.py                      # SpeechRecognizer facade/compatibility aliases
├── config.py                   # evolved typed config and profiles
├── models.py                   # result/event models
├── exceptions.py
├── audio/
│   ├── processor.py            # already shared
│   ├── devices.py              # extracted selection policy
│   ├── recorder.py             # injected capture interface
│   └── vad.py                  # mature adaptive detector/buffering
├── asr/
│   ├── engine.py               # mature decode policy/model lifecycle
│   ├── profiles.py
│   └── scheduler.py            # queue/refinement policy
├── languages/
│   ├── policy.py               # prompts/hotwords and detection
│   └── scripts.py              # script evidence/transliteration
├── text/
│   ├── normalization.py
│   ├── deduplication.py
│   └── vocabulary.py
├── runtime/
│   ├── session.py
│   ├── cancellation.py
│   └── events.py
└── logging/
    ├── logger.py               # quiet default
    └── diagnostics.py          # explicit opt-in rich diagnostics

app/
├── main.py                     # thin PyQt composition/view
└── compatibility adapters      # temporary old import paths
```

## 11. Migration steps and phase gates

1. **Phase 0 — establish baseline:** resolve the pre-existing untracked egg-info,
   create the approved branch/tag, run/preserve full regression and performance reports.
2. **Phase 1 — approve this inventory:** correct omissions before structural edits.
3. **Phase 2 — characterization:** add tests for adaptive VAD/buffering, mature decode,
   promotion, app controller cleanup and file transcription where coverage is absent.
4. **Phase 3 — API contracts:** add version exposure, config/result/event contracts and
   compatibility tests without changing the app implementation.
5. **Phase 4 — shared text/language:** move or wrap pure processing and decoding policy;
   leave old `app.*` imports as re-exporting facades.
6. **Phase 5 — shared runtime boundaries:** expose instance-owned recognition/session
   state and cancellation while preserving generation semantics.
7. **Phase 6 — mature ASR core:** adapt model provider/decode workers to callback/event
   protocols, preserving recovery, fallback, modes and metrics.
8. **Phase 7 — mature audio core:** converge recorder, capture, adaptive VAD and queue
   behavior only after characterization; retain injected backends.
9. **Phase 8 — GUI consumer:** change `SpeechController`/workers to consume the public
   API through a Qt adapter; keep all widgets outside the package.
10. **Phase 9 — translation/diagnostic extras:** make imports lazy and explicitly
    configured; align metadata with audited imports.
11. **Phase 10 — file API and examples:** add actual file transcription plus required
    English/Hindi/Hinglish/switch examples using one API.
12. **Phase 11 — installation isolation:** build a wheel, install it into a fresh venv,
    execute a consumer from outside the checkout and inspect installed dependencies.
13. **Phase 12 — full regression/performance:** compare all saved baseline metrics;
    investigate rather than relax thresholds.
14. **Phase 13 — release gate:** release only after every required environment-backed
    check has evidence.

Each phase should be a reviewable commit and preserve temporary compatibility imports.

## 12. Compatibility and extraction risks

1. **Two pipelines differ materially.** The public buffer uses a fixed RMS gate while
   the app has adaptive hysteresis, resampling, warm-up, richer queue rules, session
   generations and three modes. Replacing the latter would weaken behavior.
2. **Public config mismatch.** `SpeechConfig` does not model performance, script,
   translation, mature VAD thresholds or diagnostics; `AppConfig` does.
3. **Public model mismatch.** `TranscriptionResult` lacks mode/script/session/state and
   raw-vs-processed text needed by mature refinement.
4. **UI/controller colocation.** Importing `SpeechController` currently imports PyQt and
   translation, defeating optional core dependencies.
5. **Import-time heavy app dependencies.** `app.asr.asr_engine` imports Faster-Whisper
   and psutil directly; `app.processing.translation` imports Transformers directly.
6. **Stop semantics differ.** Public `stop()` joins workers up to two seconds; desktop
   Stop intentionally avoids joins on the UI path and invalidates generations.
7. **Language naming differs.** Package uses `None`/aliases for auto; app and reports
   mix codes and labels. Normalize internally without changing accepted inputs.
8. **Default capture differs.** Documentation/tests preserve a desktop default of
   English system audio, while public configuration defaults to microphone.
9. **Dependency metadata is incomplete for the whole app.** Extras must follow actual
   imports and should not make model, GUI or translation dependencies mandatory for a
   lightweight import.
10. **Logging philosophies differ.** Package is quiet; application eagerly creates rich
    sessions when launched. Preserve both through explicit configuration.
11. **Global diagnostics/session state.** Multiple recognizers could collide if mature
    globals are moved directly into core instead of injected per instance.
12. **Source-sensitive tests.** Some UI/template tests deliberately inspect source text;
    adapters may require behavior-preserving test evolution without weaker assertions.
13. **Platform and model variability.** SoundCard, TTS, CUDA, model downloads and Hindi
    fixture fidelity require real environment evidence; deterministic doubles cannot
    claim acoustic accuracy.
14. **Generated artifact hygiene.** Editable installs create egg-info under `src/`; its
    current untracked presence must be resolved without discarding owner work.

## 13. Tests to run after each phase

### Every commit

```bash
python -m pytest -q tests/unit tests/integration \
  --ignore=tests/integration/test_audio_generation.py
python -m pytest -q tests/stopping tests/switching
python -m pytest --collect-only -q
```

### Text/language/script phases

```bash
python -m pytest -q tests/unit/test_decoding_policy.py \
  tests/unit/test_text_processing.py tests/unit/test_live_transcript.py \
  tests/unit/test_technical_vocabulary.py tests/unit/test_technical_questions.py
```

### Runtime/audio/ASR phases

```bash
python -m pytest -q tests/integration/test_audio_pipeline.py \
  tests/unit/test_mode_comparison.py tests/stopping tests/switching
python -m pytest -q tests/unit/speakscribe
```

### Packaging/API phases

```bash
python -m build
python -m pytest -q tests/unit/speakscribe tests/unit/test_package_structure.py
python -m venv /tmp/speakscribe-clean
/tmp/speakscribe-clean/bin/python -m pip install dist/*.whl
cd /tmp && /tmp/speakscribe-clean/bin/python -c \
  "import speakscribe; from speakscribe import SpeechToText"
```

Use the equivalent `Scripts/python.exe` path on Windows. Add an installed-wheel
consumer test for `SpeechRecognizer` and `transcribe_file` when those APIs exist.

### Major extraction and release candidate

```bash
python -m pytest
python tests/run_speech_suite.py
python -m evaluation.mode_comparison \
  --manifest tests/expected/transcripts.json
python evaluation_runner.py --baseline tests/results/before_extraction.json
```

Record English/Hindi/Hinglish first-text/final latency; all profile inference metrics;
Stop, restart and switch P95; accuracy/WER; CPU/memory; passes/failures/skips/xfails.
The 120-case suite, microphone, loopback and CUDA/CPU checks require appropriate real
hardware/backends and may not be represented as passed by mocks.

## 14. Recommended first version

Keep **`0.1.0`** during review and extraction. It is already the single version in
`pyproject.toml` and correctly communicates an API-stabilization release. Add
`speakscribe.__version__` from package metadata rather than introducing another manual
version literal. Do not declare `1.0.0`, publish, or tag the first reusable release until
the full release gate has installation, consumer, language, lifecycle, logging,
dependency and performance evidence.

## Review decision requested

Before any large structural change, review these key proposals:

1. Treat `app/` as the behavioral baseline and converge it into the existing package,
   rather than replacing it with the smaller package pipeline.
2. Preserve `SpeechToText`/`SpeechConfig` during the 0.x transition while introducing
   the preferred recognizer/config/event vocabulary additively.
3. Use an iterator as the primary live API and callbacks as GUI adapters.
4. Extract in the phased order above, with compatibility facades and regression gates.
5. Resolve the pre-existing untracked `src/speakscribe.egg-info/` before claiming a
   clean baseline.

No extraction should proceed until these decisions are approved.

## Exhaustive audit companion reports

The per-file classification, exact audit totals, generated-artifact ledger, overlapping
implementation comparison, and review-required move/delete candidates are recorded in
[`repository_file_inventory.md`](repository_file_inventory.md). The capability-to-code,
test, and characterization-gap mapping is recorded in
[`functionality_test_matrix.md`](functionality_test_matrix.md). Together these reports
classify every file in the proposed audit commit; no file is classified `UNKNOWN`, and
no destructive action has been performed.
