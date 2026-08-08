"""Language-aware, deterministic TTS generation for missing validation WAVs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import wave

TEST_RANDOM_SEED = 42
MANIFEST_PATH = Path(__file__).parent / "expected" / "transcripts.json"
GENERATED_MANIFEST = Path(__file__).parent / "generated_audio_manifest.json"


@dataclass
class GenerationRecord:
    case_id: str
    audio_file: str
    audio_source: str
    tts_language: str
    voice: str
    generated_from: str
    generated: bool
    status: str
    error: str = ""


def load_generation_manifest() -> dict[str, dict]:
    if not GENERATED_MANIFEST.is_file():
        return {}
    records = json.loads(GENERATED_MANIFEST.read_text(encoding="utf-8"))["files"]
    return {record["audio_file"]: record for record in records}


def save_generation_manifest(records: dict[str, dict]) -> None:
    GENERATED_MANIFEST.write_text(
        json.dumps({"seed": TEST_RANDOM_SEED, "files": list(records.values())},
                   ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _rate_for(profile: dict) -> int:
    return {"extreme_slow": -8, "slow": -4, "normal": 0,
            "fast": 4, "very_fast": 8}[profile.get("speaking_rate", "normal")]


def _run_windows_sapi(text: str, language: str, output: Path,
                      rate: int, variation: int) -> str:
    culture_preferences = {
        "English": ["en-IN", "en-US", "en-GB"],
        "Hindi": ["hi-IN"],
        "Hinglish": ["hi-IN", "en-IN"],
    }[language]
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        text_path = directory / "text.txt"
        script_path = directory / "speak.ps1"
        text_path.write_text(text, encoding="utf-8")
        preferences = ",".join(f"'{culture}'" for culture in culture_preferences)
        output_escaped = str(output).replace("'", "''")
        text_escaped = str(text_path).replace("'", "''")
        script_path.write_text(
            "$ErrorActionPreference='Stop'\n"
            "Add-Type -AssemblyName System.Speech\n"
            "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer\n"
            f"$prefs=@({preferences})\n"
            "$voices=@($s.GetInstalledVoices() | Where-Object {$_.Enabled})\n"
            "$chosen=$null\n"
            "foreach($culture in $prefs){$matches=@($voices | Where-Object "
            "{$_.VoiceInfo.Culture.Name -eq $culture}); if($matches.Count -gt 0){"
            + "$chosen=$matches[" + str(variation) + " % $matches.Count]; break}}\n"
            "if($null -eq $chosen){throw 'No installed compatible voice for: '+($prefs -join ',')}\n"
            "$s.SelectVoice($chosen.VoiceInfo.Name)\n"
            f"$s.Rate={max(-10, min(10, rate))}\n"
            "$s.Volume=100\n"
            f"$s.SetOutputToWaveFile('{output_escaped}')\n"
            f"$s.Speak([IO.File]::ReadAllText('{text_escaped}',"
            "[Text.Encoding]::UTF8))\n"
            "$s.Dispose()\n"
            "$chosen.VoiceInfo.Name\n",
            encoding="utf-8",
        )
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script_path)],
            check=True, capture_output=True, text=True, timeout=180,
        )
        return completed.stdout.strip().splitlines()[-1]


def _run_espeak(text: str, language: str, output: Path, rate: int,
                variation: int) -> str:
    executable = shutil.which("espeak-ng") or shutil.which("espeak")
    if executable is None:
        raise RuntimeError("Neither espeak-ng nor espeak is installed")
    voice = "en-in" if language == "English" else "hi"
    words_per_minute = max(80, min(320, 175 + rate * 15))
    subprocess.run(
        [executable, "-v", voice, "-s", str(words_per_minute), "-w", str(output), text],
        check=True, capture_output=True, timeout=180,
    )
    return f"{Path(executable).name}:{voice}:variant-{variation}"


def _run_macos_say(text: str, language: str, output: Path, rate: int,
                    variation: int) -> str:
    voice = "Samantha" if language == "English" else "Lekha"
    words_per_minute = max(90, min(300, 175 + rate * 15))
    aiff = output.with_suffix(".aiff")
    subprocess.run(["say", "-v", voice, "-r", str(words_per_minute),
                    "-o", str(aiff), text], check=True, timeout=180)
    afconvert = shutil.which("afconvert")
    if afconvert is None:
        raise RuntimeError("afconvert is required to create WAV files on macOS")
    subprocess.run([afconvert, "-f", "WAVE", "-d", "LEI16", str(aiff), str(output)],
                   check=True, timeout=180)
    aiff.unlink(missing_ok=True)
    return f"macOS:{voice}:variant-{variation}"


def synthesize_clean(text: str, language: str, output: Path,
                     profile: dict, variation: int) -> str:
    """Generate speech through an OS TTS backend, entirely separate from ASR."""
    rate = _rate_for(profile)
    system = platform.system()
    if system == "Windows":
        return _run_windows_sapi(text, language, output, rate, variation)
    if system == "Darwin":
        return _run_macos_say(text, language, output, rate, variation)
    return _run_espeak(text, language, output, rate, variation)


def _read_wav(path: Path):
    import numpy as np
    from audio_pipeline import resample_audio_block

    with wave.open(str(path), "rb") as stream:
        channels, width, rate = stream.getnchannels(), stream.getsampwidth(), stream.getframerate()
        frames = stream.readframes(stream.getnframes())
    if width not in (1, 2, 4):
        raise ValueError(f"Unsupported generated sample width: {width}")
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
    audio = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    audio = (audio - 128) / 128 if width == 1 else audio / float(2 ** (width * 8 - 1))
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return resample_audio_block(audio, rate, 16_000)


def _write_wav(path: Path, audio) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(pcm.tobytes())


def apply_audio_profile(clean_path: Path, final_path: Path, profile: dict,
                        case_id: str) -> None:
    import numpy as np

    audio = _read_wav(clean_path)
    gain = 10 ** (float(profile.get("gain_db", 0)) / 20)
    audio = audio * gain
    seed = TEST_RANDOM_SEED + sum((index + 1) * ord(char)
                                  for index, char in enumerate(case_id))
    rng = np.random.default_rng(seed)
    noise = profile.get("noise", "none")
    if noise != "none":
        level = {"light_room": .003, "fan_low": .004, "fan_medium": .008,
                 "fan_high": .014, "keyboard": .01, "mouse_clicks": .008,
                 "conversation": .012, "traffic": .015, "office": .012}.get(noise, .006)
        samples = np.arange(audio.size, dtype=np.float32) / 16_000
        if noise.startswith("fan"):
            overlay = np.sin(2 * math.pi * 90 * samples) + .35 * rng.standard_normal(audio.size)
        elif noise in {"keyboard", "mouse_clicks"}:
            overlay = np.zeros(audio.size, dtype=np.float32)
            count = max(1, audio.size // (4000 if noise == "keyboard" else 8000))
            points = rng.integers(0, max(1, audio.size - 160), count)
            for point in points:
                overlay[point:point + 160] += np.hanning(160) * rng.uniform(.5, 1)
        else:
            overlay = rng.standard_normal(audio.size)
        overlay /= max(1e-6, float(np.max(np.abs(overlay))))
        audio = audio + level * overlay
    internal_pause = float(profile.get("internal_pause_seconds", 0))
    if internal_pause:
        # Insert at the quietest 20 ms frame near the center, avoiding a cut in
        # the middle of a spoken word while keeping the transformation general.
        frame = 320
        start, end = audio.size * 35 // 100, audio.size * 65 // 100
        candidates = range(start, max(start + 1, end - frame), frame)
        midpoint = min(candidates, key=lambda index: float(
            np.mean(np.square(audio[index:index + frame]), dtype=np.float64)))
        audio = np.concatenate((audio[:midpoint], np.zeros(round(16_000 * internal_pause)),
                                audio[midpoint:]))
    leading = np.zeros(round(16_000 * float(profile.get("leading_silence_seconds", 0))))
    trailing = np.zeros(round(16_000 * float(profile.get("trailing_silence_seconds", 0))))
    _write_wav(final_path, np.concatenate((leading, audio, trailing)))


def validate_wav(path: Path) -> dict[str, float | int]:
    import numpy as np

    if not path.is_file() or path.stat().st_size <= 44:
        raise ValueError("WAV is missing or empty")
    audio = _read_wav(path)
    duration = audio.size / 16_000
    rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64))) if audio.size else 0
    if duration <= .1:
        raise ValueError("WAV duration is too short")
    if rms <= 1e-6:
        raise ValueError("WAV contains no non-silent audio")
    return {"sample_rate": 16_000, "samples": int(audio.size),
            "duration_seconds": round(duration, 3), "rms": rms}


def ensure_audio(case: dict, root: Path, regenerate: bool = False) -> GenerationRecord:
    final_path = root / case["audio"]
    records = load_generation_manifest()
    prior = records.get(case["audio"])
    prior_is_generated = bool(prior and prior.get("generated") and
                              prior.get("status") == "GENERATED")
    if final_path.is_file() and not (regenerate and prior_is_generated):
        validate_wav(final_path)
        source = "synthetic" if prior_is_generated else "human"
        return GenerationRecord(case["id"], case["audio"], source, case["language"],
                                prior.get("voice", "human-recording") if prior else "human-recording",
                                "expected" if source == "synthetic" else "human", False, "READY")
    base = root / "tests/generated/base" / f"{case['id']}.wav"
    try:
        base.parent.mkdir(parents=True, exist_ok=True)
        voice = synthesize_clean(case["expected"], case["language"], base,
                                 case["audio_profile"], int(case["id"].rsplit("-", 1)[1]))
        validate_wav(base)
        apply_audio_profile(base, final_path, case["audio_profile"], case["id"])
        validate_wav(final_path)
        record = GenerationRecord(case["id"], case["audio"], "synthetic",
                                  case["language"], voice, "expected", True, "GENERATED")
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        record = GenerationRecord(case["id"], case["audio"], "synthetic",
                                  case["language"], "unavailable", "expected", False,
                                  "TTS_GENERATION_ERROR", str(exc))
    records[case["audio"]] = asdict(record)
    save_generation_manifest(records)
    return record


def generate_all(cases: list[dict], root: Path, regenerate: bool = False) -> list[GenerationRecord]:
    return [ensure_audio(case, root, regenerate) for case in cases]
