# Exhaustive repository file inventory

**Inventory date:** 2026-08-13  
**Scope rule:** every one of the 152 files in the proposed audit commit is listed below. Runtime/build artifacts are inventoried separately by pattern because a single pytest run generated more than one thousand per-test evidence files. No file was deleted, moved, renamed, or ignored in this phase.

## Audit totals

- **TOTAL FILES IN THE PROPOSED AUDIT COMMIT: 152**
- **TOTAL FILES PRESENT AT THE FINAL AUDIT SNAPSHOT: 2,276** (152 proposed tracked source/fixture/documentation files plus 2,124 ignored or untracked generated files).
- **CORE FILES: 27**
- **PUBLIC API FILES: 4**
- **INTERNAL PACKAGE MARKERS: 6**
- **APPLICATION FILES: 2**
- **TEST FILES/FIXTURE PLACEHOLDERS: 79**
- **EXAMPLE FILES: 6**
- **DEVELOPMENT TOOL FILES: 11**
- **DOCUMENTATION/LICENSE FILES: 14**
- **LEGACY COMPATIBILITY/REVIEW FILES: 3**
- **TRACKED GENERATED FILES: 0** (`evaluation/reports/.gitkeep` is an intentional directory placeholder, not generated output).
- **GENERATED FILES PRESENT: 2,124**
- **POSSIBLY UNUSED FILES: 1** (`evaluation/cases.json`; referenced by documentation but not the current runner).
- **DUPLICATE FILES: 0 proven byte/function duplicates.** There are overlapping implementations, listed below, but none is safe to delete.
- **UNKNOWN FILES: 0**

Classification is based on contents plus import, call, pytest, documentation, dynamic-import, subprocess, and path-reference searches—not filenames alone. `KEEP` means “do not destructively change in the audit phase,” not “this path is necessarily final.”

## Tracked-file inventory

| File | Purpose | Used by / evidence | Classification | Action |
|---|---|---|---|---|
| `.gitignore` | Generated/build/runtime exclusion policy | Git and contributors | DEVELOPMENT TOOL | UPDATE: add *.egg-info/, build/dist/coverage patterns |
| `LICENSE` | Repository license | Packaging and distribution | DOCUMENTATION | KEEP |
| `README.md` | Documentation/policy for . | users, developers, tests where referenced | DOCUMENTATION | KEEP/UPDATE |
| `SPEAKSCRIBE.md` | Documentation/policy for . | users, developers, tests where referenced | DOCUMENTATION | KEEP/UPDATE |
| `app/__init__.py` | Package marker/export surface | Python import system | INTERNAL | KEEP |
| `app/asr/__init__.py` | Package marker/export surface | Python import system | INTERNAL | KEEP |
| `app/asr/asr_engine.py` | Behavioral-baseline asr engine | application, evaluation and regression tests | CORE | KEEP; EXTRACT BEHIND COMPATIBILITY FACADE |
| `app/asr/language_transition.py` | Behavioral-baseline language transition | application, evaluation and regression tests | CORE | KEEP; EXTRACT BEHIND COMPATIBILITY FACADE |
| `app/asr/session_transition.py` | Behavioral-baseline session transition | application, evaluation and regression tests | CORE | KEEP; EXTRACT BEHIND COMPATIBILITY FACADE |
| `app/audio/__init__.py` | Package marker/export surface | Python import system | INTERNAL | KEEP |
| `app/audio/audio_pipeline.py` | Behavioral-baseline audio pipeline | application, evaluation and regression tests | CORE | KEEP; EXTRACT BEHIND COMPATIBILITY FACADE |
| `app/config/__init__.py` | Package marker/export surface | Python import system | INTERNAL | KEEP |
| `app/config/decoding_policy.py` | Behavioral-baseline decoding policy | application, evaluation and regression tests | CORE | KEEP; EXTRACT BEHIND COMPATIBILITY FACADE |
| `app/config/settings.py` | Behavioral-baseline settings | application, evaluation and regression tests | CORE | KEEP; EXTRACT BEHIND COMPATIBILITY FACADE |
| `app/config/technical_vocabulary.py` | Behavioral-baseline technical vocabulary | application, evaluation and regression tests | CORE | KEEP; EXTRACT BEHIND COMPATIBILITY FACADE |
| `app/final_only_main.py` | Desktop PyQt application/controller | root/example launchers and UI tests | APPLICATION | KEEP IN app; adapt to public API later |
| `app/main.py` | Desktop PyQt application/controller | root/example launchers and UI tests | APPLICATION | KEEP IN app; adapt to public API later |
| `app/processing/__init__.py` | Package marker/export surface | Python import system | INTERNAL | KEEP |
| `app/processing/live_transcript.py` | Behavioral-baseline live transcript | application, evaluation and regression tests | CORE | KEEP; EXTRACT BEHIND COMPATIBILITY FACADE |
| `app/processing/technical_questions.py` | Behavioral-baseline technical questions | application, evaluation and regression tests | CORE | KEEP; EXTRACT BEHIND COMPATIBILITY FACADE |
| `app/processing/text_processing.py` | Behavioral-baseline text processing | application, evaluation and regression tests | CORE | KEEP; EXTRACT BEHIND COMPATIBILITY FACADE |
| `app/processing/translation.py` | Behavioral-baseline translation | application, evaluation and regression tests | CORE | KEEP; EXTRACT BEHIND COMPATIBILITY FACADE |
| `app/utils/__init__.py` | Package marker/export surface | Python import system | INTERNAL | KEEP |
| `app/utils/logger.py` | Behavioral-baseline logger | application, evaluation and regression tests | CORE | KEEP; EXTRACT BEHIND COMPATIBILITY FACADE |
| `app/utils/pipeline_diagnostics.py` | Behavioral-baseline pipeline diagnostics | application, evaluation and regression tests | CORE | KEEP; EXTRACT BEHIND COMPATIBILITY FACADE |
| `data/README.md` | Documentation/policy for data | users, developers, tests where referenced | DOCUMENTATION | KEEP/UPDATE |
| `data/test_audio/README.md` | Documentation/policy for data/test_audio | users, developers, tests where referenced | DOCUMENTATION | KEEP/UPDATE |
| `docs/repository_functionality_inventory.md` | Documentation/policy for docs | users, developers, tests where referenced | DOCUMENTATION | KEEP/UPDATE |
| `docs/repository_file_inventory.md` | Exhaustive classification, generated-artifact and action ledger | cleanup review and future phases | DOCUMENTATION | KEEP/UPDATE |
| `docs/functionality_test_matrix.md` | Capability-to-implementation/test traceability | extraction and test-hardening review | DOCUMENTATION | KEEP/UPDATE |
| `evaluation/__init__.py` | Package marker/export surface | Python import system | DEVELOPMENT TOOL | KEEP |
| `evaluation/audio_generation.py` | Evaluation tool: audio generation | tests, evaluation CLI or reports | DEVELOPMENT TOOL | KEEP |
| `evaluation/cases.json` | Legacy 12-case data/test_audio manifest | data/test_audio/README.md only; not current runner | LEGACY | REVIEW; migrate/reference before any removal |
| `evaluation/evaluation_runner.py` | Evaluation tool: evaluation runner | tests, evaluation CLI or reports | DEVELOPMENT TOOL | KEEP |
| `evaluation/manifest_policy.py` | Evaluation tool: manifest policy | tests, evaluation CLI or reports | DEVELOPMENT TOOL | KEEP |
| `evaluation/mode_comparison.py` | Evaluation tool: mode comparison | tests, evaluation CLI or reports | DEVELOPMENT TOOL | KEEP |
| `evaluation/reports/.gitkeep` | Preserves intentional empty fixture/output directory | Git and path-based tools | DEVELOPMENT TOOL | KEEP |
| `evaluation/suite_logging.py` | Evaluation tool: suite logging | tests, evaluation CLI or reports | DEVELOPMENT TOOL | KEEP |
| `evaluation_runner.py` | Compatibility launcher/facade | README, users, launcher tests | LEGACY | KEEP AS LEGACY ADAPTER |
| `examples/cli_example.py` | Runnable cli example example | SPEAKSCRIBE.md/README or manual users | EXAMPLE | KEEP; REORGANIZE AFTER COMPATIBILITY REVIEW |
| `examples/live_update_main.py` | Runnable live update main example | SPEAKSCRIBE.md/README or manual users | EXAMPLE | KEEP; REORGANIZE AFTER COMPATIBILITY REVIEW |
| `examples/performance_comparison_template.py` | Runnable performance comparison template example | SPEAKSCRIBE.md/README or manual users | EXAMPLE | KEEP; REORGANIZE AFTER COMPATIBILITY REVIEW |
| `examples/pyqt_recording_panel.py` | Runnable pyqt recording panel example | SPEAKSCRIBE.md/README or manual users | EXAMPLE | KEEP; REORGANIZE AFTER COMPATIBILITY REVIEW |
| `examples/simple_example.py` | Runnable simple example example | SPEAKSCRIBE.md/README or manual users | EXAMPLE | KEEP; REORGANIZE AFTER COMPATIBILITY REVIEW |
| `examples/tkinter_example.py` | Runnable tkinter example example | SPEAKSCRIBE.md/README or manual users | EXAMPLE | KEEP; REORGANIZE AFTER COMPATIBILITY REVIEW |
| `main.py` | Compatibility launcher/facade | README, users, launcher tests | LEGACY | KEEP AS LEGACY ADAPTER |
| `pyproject.toml` | Build metadata, dependencies and version 0.1.0 | pip/setuptools/build | DEVELOPMENT TOOL | KEEP/UPDATE IN PACKAGING PHASE |
| `pytest.ini` | Pytest discovery and import paths | pytest | DEVELOPMENT TOOL | KEEP |
| `requirements.txt` | Full application/test dependency environment | developers and CI | DEVELOPMENT TOOL | KEEP; reconcile extras later |
| `scripts/README.md` | Documentation/policy for scripts | users, developers, tests where referenced | DOCUMENTATION | KEEP/UPDATE |
| `src/speakscribe/__init__.py` | Package marker/export surface | Python import system | PUBLIC API | KEEP |
| `src/speakscribe/audio/__init__.py` | Package marker/export surface | Python import system | CORE | KEEP |
| `src/speakscribe/audio/microphone.py` | Reusable library microphone | public service, package internals and library tests | CORE | KEEP |
| `src/speakscribe/audio/processor.py` | Reusable library processor | public service, package internals and library tests | CORE | KEEP |
| `src/speakscribe/audio/recorder.py` | Reusable library recorder | public service, package internals and library tests | CORE | KEEP |
| `src/speakscribe/config.py` | Reusable library config | public service, package internals and library tests | PUBLIC API | KEEP |
| `src/speakscribe/exceptions.py` | Reusable library exceptions | public service, package internals and library tests | PUBLIC API | KEEP |
| `src/speakscribe/logging/__init__.py` | Package marker/export surface | Python import system | CORE | KEEP |
| `src/speakscribe/logging/logger.py` | Reusable library logger | public service, package internals and library tests | CORE | KEEP |
| `src/speakscribe/models.py` | Reusable library models | public service, package internals and library tests | PUBLIC API | KEEP |
| `src/speakscribe/services/__init__.py` | Package marker/export surface | Python import system | CORE | KEEP |
| `src/speakscribe/services/speech_service.py` | Reusable library speech service | public service, package internals and library tests | CORE | KEEP |
| `src/speakscribe/transcription/__init__.py` | Package marker/export surface | Python import system | CORE | KEEP |
| `src/speakscribe/transcription/base.py` | Reusable library base | public service, package internals and library tests | CORE | KEEP |
| `src/speakscribe/transcription/engine.py` | Reusable library engine | public service, package internals and library tests | CORE | KEEP |
| `src/speakscribe/transcription/streaming.py` | Reusable library streaming | public service, package internals and library tests | CORE | KEEP |
| `src/speakscribe/utils/__init__.py` | Package marker/export surface | Python import system | CORE | KEEP |
| `src/speakscribe/utils/helpers.py` | Reusable library helpers | public service, package internals and library tests | CORE | KEEP |
| `tests/FAILURE_ANALYSIS.md` | Documentation/policy for tests | users, developers, tests where referenced | DOCUMENTATION | KEEP/UPDATE |
| `tests/README.md` | Documentation/policy for tests | users, developers, tests where referenced | DOCUMENTATION | KEEP/UPDATE |
| `tests/TESTING_PHILOSOPHY.md` | Documentation/policy for tests | users, developers, tests where referenced | DOCUMENTATION | KEEP/UPDATE |
| `tests/__init__.py` | Package marker/export surface | Python import system | TEST | KEEP |
| `tests/conftest.py` | Pytest hooks/fixtures | pytest test tree | TEST | KEEP |
| `tests/expected/transcripts.json` | Test manifest/provenance data | pytest, evaluation and fixture tools | TEST | KEEP |
| `tests/fixtures/__init__.py` | Package marker/export surface | Python import system | TEST | KEEP |
| `tests/fixtures/metrics.py` | Test support/runner: metrics | pytest or documented test command | TEST | KEEP |
| `tests/fixtures/pytest_observability.py` | Test support/runner: pytest observability | pytest or documented test command | TEST | KEEP |
| `tests/generate_test_audio.py` | Test support/runner: generate test audio | pytest or documented test command | TEST | KEEP |
| `tests/generated_audio_manifest.json` | Test manifest/provenance data | pytest, evaluation and fixture tools | TEST | KEEP |
| `tests/integration/__init__.py` | Package marker/export surface | Python import system | TEST | KEEP |
| `tests/integration/test_audio_generation.py` | Behavioral coverage: audio generation | pytest; protects production behavior | TEST | KEEP |
| `tests/integration/test_audio_pipeline.py` | Behavioral coverage: audio pipeline | pytest; protects production behavior | TEST | KEEP |
| `tests/integration/test_run_speech_suite.py` | Behavioral coverage: run speech suite | pytest; protects production behavior | TEST | KEEP |
| `tests/regression/__init__.py` | Package marker/export surface | Python import system | TEST | KEEP |
| `tests/run_speech_suite.py` | Test support/runner: run speech suite | pytest or documented test command | TEST | KEEP |
| `tests/speech/__init__.py` | Package marker/export surface | Python import system | TEST | KEEP |
| `tests/speech/test_transcription.py` | Behavioral coverage: transcription | pytest; protects production behavior | TEST | KEEP |
| `tests/speech_cases/01_normal/.gitkeep` | Preserves intentional empty fixture/output directory | Git and path-based tools | TEST | KEEP |
| `tests/speech_cases/02_fast/.gitkeep` | Preserves intentional empty fixture/output directory | Git and path-based tools | TEST | KEEP |
| `tests/speech_cases/03_slow/.gitkeep` | Preserves intentional empty fixture/output directory | Git and path-based tools | TEST | KEEP |
| `tests/speech_cases/04_short/.gitkeep` | Preserves intentional empty fixture/output directory | Git and path-based tools | TEST | KEEP |
| `tests/speech_cases/05_long/.gitkeep` | Preserves intentional empty fixture/output directory | Git and path-based tools | TEST | KEEP |
| `tests/speech_cases/06_multi_sentence/.gitkeep` | Preserves intentional empty fixture/output directory | Git and path-based tools | TEST | KEEP |
| `tests/speech_cases/07_pauses/.gitkeep` | Preserves intentional empty fixture/output directory | Git and path-based tools | TEST | KEEP |
| `tests/speech_cases/08_micro_pauses/.gitkeep` | Preserves intentional empty fixture/output directory | Git and path-based tools | TEST | KEEP |
| `tests/speech_cases/09_long_pause/.gitkeep` | Preserves intentional empty fixture/output directory | Git and path-based tools | TEST | KEEP |
| `tests/speech_cases/10_low_volume/.gitkeep` | Preserves intentional empty fixture/output directory | Git and path-based tools | TEST | KEEP |
| `tests/speech_cases/11_loud/.gitkeep` | Preserves intentional empty fixture/output directory | Git and path-based tools | TEST | KEEP |
| `tests/speech_cases/12_background_noise/.gitkeep` | Preserves intentional empty fixture/output directory | Git and path-based tools | TEST | KEEP |
| `tests/speech_cases/13_fan_noise/.gitkeep` | Preserves intentional empty fixture/output directory | Git and path-based tools | TEST | KEEP |
| `tests/speech_cases/14_office_noise/.gitkeep` | Preserves intentional empty fixture/output directory | Git and path-based tools | TEST | KEEP |
| `tests/speech_cases/15_accent/.gitkeep` | Preserves intentional empty fixture/output directory | Git and path-based tools | TEST | KEEP |
| `tests/speech_cases/16_numbers/.gitkeep` | Preserves intentional empty fixture/output directory | Git and path-based tools | TEST | KEEP |
| `tests/speech_cases/17_datetime/.gitkeep` | Preserves intentional empty fixture/output directory | Git and path-based tools | TEST | KEEP |
| `tests/speech_cases/18_names/.gitkeep` | Preserves intentional empty fixture/output directory | Git and path-based tools | TEST | KEEP |
| `tests/speech_cases/19_technical/.gitkeep` | Preserves intentional empty fixture/output directory | Git and path-based tools | TEST | KEEP |
| `tests/speech_cases/20_acronyms/.gitkeep` | Preserves intentional empty fixture/output directory | Git and path-based tools | TEST | KEEP |
| `tests/speech_cases/21_repository/.gitkeep` | Preserves intentional empty fixture/output directory | Git and path-based tools | TEST | KEEP |
| `tests/speech_cases/22_repeated/.gitkeep` | Preserves intentional empty fixture/output directory | Git and path-based tools | TEST | KEEP |
| `tests/speech_cases/23_homophones/.gitkeep` | Preserves intentional empty fixture/output directory | Git and path-based tools | TEST | KEEP |
| `tests/speech_cases/24_switch/.gitkeep` | Preserves intentional empty fixture/output directory | Git and path-based tools | TEST | KEEP |
| `tests/speech_cases/25_multi_switch/.gitkeep` | Preserves intentional empty fixture/output directory | Git and path-based tools | TEST | KEEP |
| `tests/speech_cases/26_natural_hinglish/.gitkeep` | Preserves intentional empty fixture/output directory | Git and path-based tools | TEST | KEEP |
| `tests/speech_cases/27_english_in_hindi/.gitkeep` | Preserves intentional empty fixture/output directory | Git and path-based tools | TEST | KEEP |
| `tests/speech_cases/28_punctuation/.gitkeep` | Preserves intentional empty fixture/output directory | Git and path-based tools | TEST | KEEP |
| `tests/speech_cases/29_silence/.gitkeep` | Preserves intentional empty fixture/output directory | Git and path-based tools | TEST | KEEP |
| `tests/speech_cases/30_combined/.gitkeep` | Preserves intentional empty fixture/output directory | Git and path-based tools | TEST | KEEP |
| `tests/stopping/README.md` | Documentation/policy for tests/stopping | users, developers, tests where referenced | DOCUMENTATION | KEEP/UPDATE |
| `tests/stopping/__init__.py` | Package marker/export surface | Python import system | TEST | KEEP |
| `tests/stopping/test_stop_complex.py` | Behavioral coverage: stop complex | pytest; protects production behavior | TEST | KEEP |
| `tests/stopping/test_stop_light.py` | Behavioral coverage: stop light | pytest; protects production behavior | TEST | KEEP |
| `tests/switching/README.md` | Documentation/policy for tests/switching | users, developers, tests where referenced | DOCUMENTATION | KEEP/UPDATE |
| `tests/switching/__init__.py` | Package marker/export surface | Python import system | TEST | KEEP |
| `tests/switching/complex/test_concurrency_and_audio.py` | Behavioral coverage: concurrency and audio | pytest; protects production behavior | TEST | KEEP |
| `tests/switching/conftest.py` | Pytest hooks/fixtures | pytest test tree | TEST | KEEP |
| `tests/switching/light/test_basic_switch.py` | Behavioral coverage: basic switch | pytest; protects production behavior | TEST | KEEP |
| `tests/switching/light/test_switch_latency.py` | Behavioral coverage: switch latency | pytest; protects production behavior | TEST | KEEP |
| `tests/switching/light/test_switch_state.py` | Behavioral coverage: switch state | pytest; protects production behavior | TEST | KEEP |
| `tests/switching/negative/test_failure_and_stress.py` | Behavioral coverage: failure and stress | pytest; protects production behavior | TEST | KEEP |
| `tests/switching/reporting.py` | Test support/runner: reporting | pytest or documented test command | TEST | KEEP |
| `tests/switching/support.py` | Test support/runner: support | pytest or documented test command | TEST | KEEP |
| `tests/unit/__init__.py` | Package marker/export surface | Python import system | TEST | KEEP |
| `tests/unit/speakscribe/test_audio.py` | Behavioral coverage: audio | pytest; protects production behavior | TEST | KEEP |
| `tests/unit/speakscribe/test_package.py` | Behavioral coverage: package | pytest; protects production behavior | TEST | KEEP |
| `tests/unit/speakscribe/test_speech_service.py` | Behavioral coverage: speech service | pytest; protects production behavior | TEST | KEEP |
| `tests/unit/test_decoding_policy.py` | Behavioral coverage: decoding policy | pytest; protects production behavior | TEST | KEEP |
| `tests/unit/test_evaluation.py` | Behavioral coverage: evaluation | pytest; protects production behavior | TEST | KEEP |
| `tests/unit/test_launcher_templates.py` | Behavioral coverage: launcher templates | pytest; protects production behavior | TEST | KEEP |
| `tests/unit/test_live_status_template.py` | Behavioral coverage: live status template | pytest; protects production behavior | TEST | KEEP |
| `tests/unit/test_live_transcript.py` | Behavioral coverage: live transcript | pytest; protects production behavior | TEST | KEEP |
| `tests/unit/test_logger.py` | Behavioral coverage: logger | pytest; protects production behavior | TEST | KEEP |
| `tests/unit/test_manifest_policy.py` | Behavioral coverage: manifest policy | pytest; protects production behavior | TEST | KEEP |
| `tests/unit/test_mode_comparison.py` | Behavioral coverage: mode comparison | pytest; protects production behavior | TEST | KEEP |
| `tests/unit/test_package_structure.py` | Behavioral coverage: package structure | pytest; protects production behavior | TEST | KEEP |
| `tests/unit/test_pipeline_diagnostics.py` | Behavioral coverage: pipeline diagnostics | pytest; protects production behavior | TEST | KEEP |
| `tests/unit/test_processing_before_final.py` | Behavioral coverage: processing before final | pytest; protects production behavior | TEST | KEEP |
| `tests/unit/test_pytest_observability.py` | Behavioral coverage: pytest observability | pytest; protects production behavior | TEST | KEEP |
| `tests/unit/test_suite_logging.py` | Behavioral coverage: suite logging | pytest; protects production behavior | TEST | KEEP |
| `tests/unit/test_technical_questions.py` | Behavioral coverage: technical questions | pytest; protects production behavior | TEST | KEEP |
| `tests/unit/test_technical_vocabulary.py` | Behavioral coverage: technical vocabulary | pytest; protects production behavior | TEST | KEEP |
| `tests/unit/test_text_processing.py` | Behavioral coverage: text processing | pytest; protects production behavior | TEST | KEEP |
| `tests/unit/test_validation_manifest.py` | Behavioral coverage: validation manifest | pytest; protects production behavior | TEST | KEEP |

## Generated and untracked artifact inventory

| Paths observed | Count | Classification | References / provenance | Proposed action |
|---|---:|---|---|---|
| `**/__pycache__/*.pyc` | 87 | GENERATED | Python/pytest bytecode caches | Remove after approval; already covered by `__pycache__/` and `*.py[cod]`. |
| `.pytest_cache/**` | 4 | GENERATED | Pytest collection cache | Remove after approval; already covered by `.pytest_cache/`. |
| `test_logs/**` | 2,026 | GENERATED | Central pytest observability output from baseline commands | Preserve outside Git as run evidence or archive externally; directory is already ignored. |
| `test_reports/*.xlsx` | 2 | GENERATED | Switching-suite workbook | Preserve/archive as baseline evidence; pattern is already ignored. |
| `src/speakscribe.egg-info/**` | 5 | GENERATED | Editable-install metadata, present before this audit | Remove after approval and add `*.egg-info/` to `.gitignore`; never package as source. |

These counts reconcile to 2,124 generated/untracked files. Intentional tracked `.gitkeep` files and JSON fixture manifests are not counted as generated artifacts. No `logs/`, `debug_audio/`, `build/`, `dist/`, coverage output, shell script, PowerShell script, batch script, YAML, CFG, UI asset, or template directory was physically present during this audit.

## Overlapping implementations (not proven duplicate files)

| Capability | Mature application implementation | Reusable-package implementation | Why neither may be deleted now |
|---|---|---|---|
| Audio capture | `app.audio.audio_pipeline.AudioCaptureWorker`, `select_capture_device` | `speakscribe.audio.SoundCardRecorder`, `capture_device` | App adds batching, warm-up, resampling and discontinuity handling; package adds injected recorder abstraction. |
| VAD/buffering | `EnergySpeechDetector`, `SpeechBufferWorker` | `StreamingBuffer` | Adaptive hysteresis/generation queues differ from fixed-RMS public buffering. |
| ASR/model lifecycle | `WhisperEngine`, `WhisperModelProvider`, ASR workers | `FasterWhisperEngine`, `BaseTranscriptionEngine` | App preserves three profiles, script policy, recovery and stale-result handling; package preserves backend injection and lightweight imports. |
| Stop/lifecycle | `LiveSessionBoundary`, `SpeechController.stop` | `SpeechToText.stop/close` | Desktop Stop is non-blocking and generation-aware; package joins instance workers and offers context management. |
| Logging | `app.utils.logger` | `speakscribe.logging` | Rich app sessions and quiet library defaults are both valid behaviors. |
| Text/script handling | `app.processing.text_processing` | No package equivalent | Mature behavior must be extracted, not dropped. |
| Language switching | `RecognitionState`, controller switch | No public package equivalent | Tested generation semantics must become reusable before GUI conversion. |

## Proposed destructive and move actions (review required)

No tracked deletion is proposed in the next phase. The following are the complete candidates; execute none until approved.

### Generated deletion candidate A

- **Current paths:** `**/__pycache__/`, `.pytest_cache/`.
- **Purpose:** interpreter/test caches.
- **References:** no source references; recreated automatically.
- **Tests:** none consume them.
- **Proposed action:** delete generated contents only.
- **Reason:** generated and ignored.
- **Risk:** negligible; first subsequent run is uncached.

### Generated deletion candidate B

- **Current path:** `src/speakscribe.egg-info/`.
- **Purpose:** setuptools editable-install metadata.
- **References:** not imported or referenced by source/tests/docs.
- **Tests:** packaging recreates it.
- **Proposed action:** delete generated directory and add `*.egg-info/` to `.gitignore`.
- **Reason:** generated, untracked build metadata.
- **Risk:** negligible after installation tests; an editable install may recreate it.

### Generated evidence candidate C

- **Current paths:** `test_logs/`, `test_reports/`.
- **Purpose:** test diagnostics and switching workbook.
- **References:** test philosophy intentionally creates them, but runtime does not read old sessions.
- **Tests:** observability/reporter tests verify creation, not retention of these exact sessions.
- **Proposed action:** archive baseline summaries externally, then delete local generated sessions if approved.
- **Reason:** ignored generated evidence, not source.
- **Risk:** loss of audit evidence if removed before baseline results are preserved.

### Tracked review candidate D

- **Current path:** `evaluation/cases.json`.
- **Purpose:** legacy 12-case manifest for `data/test_audio/`.
- **References:** `data/test_audio/README.md`; no current Python caller found.
- **Tests:** current evaluation and manifest tests use `tests/expected/transcripts.json`, not this file.
- **Proposed action:** KEEP now; later either restore a documented legacy/manual runner or migrate its unique 12 cases into the governed manifest before considering deletion.
- **Reason:** possibly unused but still documented and contains unique expected transcripts.
- **Risk:** deleting now could lose a manual evaluation contract and sample-audio naming policy.

### Proposed example moves (no content deletion)

| Current path | Proposed path | References/tests to update first | Risk |
|---|---|---|---|
| `examples/simple_example.py` | `examples/basic/live_microphone_once.py` | `SPEAKSCRIBE.md` | Low; preserve old wrapper temporarily. |
| `examples/cli_example.py` | `examples/basic/live_microphone.py` | `SPEAKSCRIBE.md` | Low; preserve documented invocation. |
| `examples/tkinter_example.py` | `examples/gui/tkinter_app.py` | package-boundary test and docs | Medium; source-location assertion exists. |
| `examples/pyqt_recording_panel.py` | `examples/gui/pyqt_recording_panel.py` | package-boundary test and docs | Medium; source-location assertion exists. |
| `examples/live_update_main.py` | `examples/gui/templates/live_update_main.py` | launcher-template tests and README | Medium; historical module invocation must remain. |
| `examples/performance_comparison_template.py` | `examples/gui/templates/performance_comparison.py` | launcher/template tests and README | Medium; direct checkout execution and root-path bootstrap must remain. |

`app/main.py` and `app/final_only_main.py` should remain under `app/`, not be mislabeled as examples. They are the existing supported desktop application. Their internal controller should consume the public library only after the mature runtime is extracted and characterized.

## Decision gate

Approval is requested for classification and for the proposed non-destructive next phase: baseline preservation plus characterization/public-contract tests. No generated cleanup, tracked deletion, example move, or core extraction should occur before that review.
