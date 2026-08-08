from pathlib import Path

import pytest

from tests import audio_generation


CASE = {
    "id": "CASE-01", "audio": "speech_cases/case.wav", "expected": "Hello",
    "language": "English", "audio_profile": {"speaking_rate": "normal"},
}


def test_existing_unmanaged_audio_is_preserved_as_human(monkeypatch, tmp_path):
    audio = tmp_path / CASE["audio"]
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"human")
    monkeypatch.setattr(audio_generation, "GENERATED_MANIFEST", tmp_path / "generated.json")
    monkeypatch.setattr(audio_generation, "validate_wav", lambda _path: {"duration": 1})
    record = audio_generation.ensure_audio(CASE, tmp_path)
    assert record.audio_source == "human"
    assert record.status == "READY"
    assert audio.read_bytes() == b"human"


def test_missing_audio_is_generated_and_recorded_as_synthetic(monkeypatch, tmp_path):
    monkeypatch.setattr(audio_generation, "GENERATED_MANIFEST", tmp_path / "generated.json")
    monkeypatch.setattr(audio_generation, "validate_wav", lambda _path: {"duration": 1})

    def synthesize(_text, _language, output, _profile, _variation):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"base")
        return "test-voice"

    def transform(_base, final, _profile, _case_id):
        final.parent.mkdir(parents=True, exist_ok=True)
        final.write_bytes(b"derived")

    monkeypatch.setattr(audio_generation, "synthesize_clean", synthesize)
    monkeypatch.setattr(audio_generation, "apply_audio_profile", transform)
    record = audio_generation.ensure_audio(CASE, tmp_path)
    assert record.audio_source == "synthetic"
    assert record.status == "GENERATED"
    assert (tmp_path / CASE["audio"]).read_bytes() == b"derived"
    assert "synthetic" in audio_generation.GENERATED_MANIFEST.read_text(encoding="utf-8")


def test_tts_failure_is_not_classified_as_asr_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(audio_generation, "GENERATED_MANIFEST", tmp_path / "generated.json")
    monkeypatch.setattr(
        audio_generation, "synthesize_clean",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("no Hindi voice")),
    )
    record = audio_generation.ensure_audio(CASE, tmp_path)
    assert record.status == "TTS_GENERATION_ERROR"
    assert "no Hindi voice" in record.error


def test_windows_voice_preferences_protect_devanagari_and_allow_latin_hinglish():
    assert audio_generation.windows_voice_preferences("Hindi", "आज काम करो") == ["hi-IN"]
    assert audio_generation.windows_voice_preferences("Hinglish", "आज deploy karo") == ["hi-IN"]
    assert audio_generation.windows_voice_preferences("Hinglish", "Aaj deploy karo") == [
        "hi-IN", "en-IN", "en-US", "en-GB",
    ]


def test_windows_tts_failure_includes_powershell_detail_and_install_hint(
        monkeypatch, tmp_path):
    class FailedProcess:
        returncode = 1
        stdout = ""
        stderr = "No installed compatible voice for: hi-IN"

    monkeypatch.setattr(audio_generation.subprocess, "run", lambda *_args, **_kwargs: FailedProcess())
    with pytest.raises(RuntimeError) as failure:
        audio_generation._run_windows_sapi(
            "आज काम पूरा करना है", "Hindi", tmp_path / "hindi.wav", 0, 0,
        )
    message = str(failure.value)
    assert "No installed compatible voice for: hi-IN" in message
    assert "Add-WindowsCapability" in message
