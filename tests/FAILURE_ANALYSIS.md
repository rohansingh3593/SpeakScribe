# Hindi `*-02` failure analysis

## Supplied baseline

The supplied 2026-08-09 run selected 30 cases and reported 19 failures, all variation
`02` cases dominated by Hindi or Hindi/English code switching. Across those 19 failures,
the reported mean similarity was **31.33%** and mean WER was **0.7188**. CUDA could not
initialize, so inference fell back to CPU; that explains latency but not the systematic
language-specific substitutions.

## Verified common causes

1. **Synthetic Hindi fixture fidelity.** Windows selected the legacy Hindi SAPI voice
   before neural synthesis. The clustering on the same language/variation and phonetic
   spellings such as `अलग` → `अलक` are consistent with the recognizer receiving
   systematically accented synthetic speech. Generator version 3 now uses neural Hindi
   voices first and rebuilds managed version-2 assets. Human recordings are untouched.
2. **Misuse of Whisper `initial_prompt`.** The final Hindi decode received an English
   instruction paragraph plus every technical vocabulary item. Faster-Whisper treats
   this parameter as preceding transcript, not as a system instruction. Debug evidence
   showed prompt-seeded repetitive output and fallback decoding. Pinned Hindi now gets
   no synthetic prompt; genuine prior transcript context remains allowed.
3. **Overconfident root-cause labels.** Offline evaluation submits the complete WAV with
   `vad_filter=False`, so a pause feature alone does not prove VAD loss. VAD/chunk loss
   is now reported only when dropped chunks are observed, and partial merging only when
   duplicated final words are observed.

## Retest requirements

The environment used for this change does not provide the Windows neural fixture path
or Faster-Whisper model runtime, so no fabricated after-score is reported. On Windows,
rerun generation first so managed version-2 WAVs are replaced, then rerun the 19 cases,
all Hindi cases, related Hinglish cases, and finally the complete manifest. Preserve the
previous JSON and use `--baseline` to measure similarity, WER, latency, and regressions.
Any remaining failures must stay visible; do not change transcripts or thresholds.
