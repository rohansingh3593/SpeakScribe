# Prerecorded evaluation audio

Place PCM WAV recordings here using the filenames in `evaluation/cases.json`.
Recordings are intentionally not synthesized or fabricated: accuracy numbers are
meaningful only when they come from the target speakers, microphones, accents,
speeds, pauses, and background-noise conditions.

WAV files may be mono or stereo and 8-, 16-, or 32-bit PCM at any sample rate.
The evaluator converts them to mono 16 kHz using the application audio path.
