from queue import Queue
from threading import Event
from types import SimpleNamespace
import warnings

import numpy as np

import app.audio.audio_pipeline as audio_pipeline
from app.audio.audio_pipeline import (
    ASRJob, AudioCaptureWorker, EnergySpeechDetector, SpeechBufferWorker,
    audio_normalization_gain, audio_statistics, prepare_audio_for_asr,
    resample_audio_block, select_capture_device,
)
from app.config.settings import AppConfig
from app.utils.logger import configure_logging


def test_energy_detector_hysteresis_and_reset():
    detector = EnergySpeechDetector(AppConfig(
        speech_threshold=0.012, silence_threshold=0.008, speech_start_frames=1,
        adaptive_vad_enabled=False,
    ))
    assert not detector.classify(np.zeros(480, dtype=np.float32))[0]
    assert detector.classify(np.full(480, 0.02, dtype=np.float32))[0]
    assert detector.classify(np.full(480, 0.009, dtype=np.float32))[0]
    detector.reset()
    assert not detector.classify(np.full(480, 0.009, dtype=np.float32))[0]


def test_default_detector_accepts_quiet_laptop_microphone_speech():
    detector = EnergySpeechDetector(AppConfig())
    frame = np.full(480, 0.0035, dtype=np.float32)
    assert not detector.classify(frame)[0]
    assert not detector.classify(frame)[0]
    assert detector.classify(frame)[0]


def test_adaptive_detector_accepts_voice_below_fixed_threshold():
    detector = EnergySpeechDetector(AppConfig(speech_start_frames=3))
    quiet_voice = np.full(480, 0.0008, dtype=np.float32)
    assert not detector.classify(quiet_voice)[0]
    assert not detector.classify(quiet_voice)[0]
    assert detector.classify(quiet_voice)[0]
    assert detector.effective_start_threshold < detector.config.speech_threshold
    # A low adaptive attack threshold may start soft speech, but the release
    # gate must still reject the persistent ~0.0005 room noise seen in the
    # Hindi diagnostic session instead of sticking in speech for 15 seconds.
    assert not detector.classify(np.full(480, 0.0006, dtype=np.float32))[0]
    assert detector.effective_silence_threshold == detector.config.silence_threshold


def test_adaptive_detector_releases_room_noise_after_normal_hindi_speech():
    detector = EnergySpeechDetector(AppConfig(speech_start_frames=3))
    speech = np.full(480, 0.02, dtype=np.float32)
    assert [detector.classify(speech)[0] for _ in range(3)] == [False, False, True]
    assert not detector.classify(np.full(480, 0.0005, dtype=np.float32))[0]


def test_quiet_speech_is_centered_and_safely_amplified_for_whisper():
    time_axis = np.linspace(0, 4 * np.pi, 1600, dtype=np.float32)
    quiet_speech = 0.02 + 0.01 * np.sin(time_axis)
    prepared = prepare_audio_for_asr(quiet_speech)
    assert abs(float(np.mean(prepared))) < 1e-5
    assert 0.14 < float(np.max(np.abs(prepared))) < 0.16
    assert float(np.max(np.abs(prepared))) <= 1.0


def test_normalization_does_not_turn_silence_or_tiny_noise_into_speech():
    silence = np.zeros(1600, dtype=np.float32)
    tiny_noise = np.full(1600, 0.0005, dtype=np.float32)
    assert audio_normalization_gain(silence) == 1.0
    assert audio_normalization_gain(tiny_noise) == 1.0
    assert np.max(np.abs(prepare_audio_for_asr(tiny_noise))) == 0.0


def test_normalization_gain_is_bounded_for_low_volume_capture():
    low_volume = np.array([-0.005, 0.005], dtype=np.float32)
    assert audio_normalization_gain(low_volume) == 15.0
    assert np.isclose(np.max(np.abs(prepare_audio_for_asr(low_volume))), 0.075)


def test_native_48khz_capture_is_downsampled_to_16khz():
    source = np.arange(1440, dtype=np.float32)
    result = resample_audio_block(source, 48_000, 16_000)
    assert result.shape == (480,)
    assert np.isclose(result[0], 1.0)


def test_audio_statistics_expose_silence_and_invalid_samples():
    silence = audio_statistics(np.zeros(480, dtype=np.float32))
    assert silence == {
        "samples": 480, "rms": 0.0, "peak": 0.0, "mean": 0.0,
        "zero_ratio": 1.0, "finite": True,
    }
    invalid = audio_statistics(np.array([0.0, np.nan], dtype=np.float32))
    assert invalid["finite"] is False


def test_capture_batches_backend_reads_and_keeps_frames_after_discontinuity(
        tmp_path, monkeypatch):
    class BackendWarning(RuntimeWarning):
        pass

    stop = Event()

    class Recorder:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def record(self, numframes):
            warnings.warn("data discontinuity in recording", BackendWarning)
            stop.set()
            return np.full((numframes, 1), 0.01, dtype=np.float32)

    microphone = SimpleNamespace(
        name="Regression microphone",
        recorder=lambda **_: Recorder(),
    )
    speaker = SimpleNamespace(name="Regression speaker")
    modules = {
        "soundcard": SimpleNamespace(
            default_speaker=lambda: speaker,
            default_microphone=lambda: microphone,
            get_microphone=lambda **_: microphone,
        ),
        "soundcard.mediafoundation": SimpleNamespace(SoundcardRuntimeWarning=BackendWarning),
    }
    monkeypatch.setattr(audio_pipeline, "import_module", modules.__getitem__)
    session = configure_logging(logs_root=tmp_path)
    output = Queue(maxsize=10)
    errors = []

    AudioCaptureWorker(AppConfig(capture_warmup_blocks=0), output, stop, errors.append).run()

    assert not errors
    assert output.qsize() == 4
    audio_log = (session.directory / "modules/audio.log").read_text(encoding="utf-8")
    assert "Audio capture active" in audio_log
    assert "data discontinuity; capture continues" in audio_log


def test_capture_source_matches_legacy_speaker_loopback_and_physical_microphone():
    speaker = SimpleNamespace(name="Speakers")
    loopback = SimpleNamespace(name="Loopback")
    microphone = SimpleNamespace(name="Microphone")
    calls = []
    soundcard = SimpleNamespace(
        default_speaker=lambda: speaker,
        default_microphone=lambda: microphone,
        get_microphone=lambda **kwargs: calls.append(kwargs) or loopback,
    )

    assert select_capture_device(soundcard, "loopback") == (
        loopback, "speaker-loopback:Speakers", None)
    assert calls == [{"id": "Speakers", "include_loopback": True}]
    assert select_capture_device(soundcard, "microphone") == (
        microphone, "microphone:Microphone", 1)


def test_default_capture_settings_use_physical_microphone_for_spoken_transcription():
    config = AppConfig()
    assert config.capture_source == "microphone"
    assert config.capture_sample_rate == 16_000
    assert config.capture_warmup_blocks == 3
    assert config.capture_warmup_ms == 100
    assert config.min_partial_duration == 0.80
    assert config.min_partial_speech_duration == 0.30
    assert config.partial_interval == 0.40


def test_comparison_backpressure_replaces_stale_final_without_blocking_vad():
    output = Queue(maxsize=1)
    old = ASRJob(np.ones(10, dtype=np.float32), True, 1, 0.0)
    newest = ASRJob(np.ones(10, dtype=np.float32), True, 2, 0.0)
    output.put(old)
    worker = SpeechBufferWorker(
        AppConfig(asr_keep_latest_final=True), Queue(), output, Event())

    worker._submit(newest)

    assert output.get_nowait() is newest


def test_utterance_ids_are_namespaced_by_generation_to_preserve_ui_history():
    worker = SpeechBufferWorker(AppConfig(), Queue(), Queue(), Event())
    first_session = worker._next_utterance_id(3)
    second_utterance = worker._next_utterance_id(3)
    next_session_worker = SpeechBufferWorker(AppConfig(), Queue(), Queue(), Event())
    restarted = next_session_worker._next_utterance_id(4)

    assert first_session == 3_000_001
    assert second_utterance == 3_000_002
    assert restarted == 4_000_001
    assert len({first_session, second_utterance, restarted}) == 3


def test_audio_preparation_can_report_gain_without_changing_samples():
    audio = np.linspace(-0.03, 0.04, 1600, dtype=np.float32)

    baseline = prepare_audio_for_asr(audio)
    prepared, gain = prepare_audio_for_asr(audio, return_gain=True)

    assert np.array_equal(prepared, baseline)
    assert gain == audio_normalization_gain(
        audio - np.mean(audio, dtype=np.float64))
