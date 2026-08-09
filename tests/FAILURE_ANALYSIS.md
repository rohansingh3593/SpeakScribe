# Hindi `*-02` failure analysis

## Supplied baseline

The supplied 2026-08-09 run selected 30 cases and reported 19 failures, all variation
`02` cases dominated by Hindi or Hindi/English code switching. Across those 19 failures,
the reported mean similarity was **31.33%** and mean WER was **0.7188**. CUDA could not
initialize, so inference fell back to CPU; that explains latency but not the systematic
language-specific substitutions.

A follow-up run after the neural-fixture/prompt correction reported **15 failures and
15 passes** in the selected Hindi group. The remaining 15 failures averaged **36.10%**
similarity and **0.6550** WER. Compared with the original failing set, that is +4.77
similarity points and -0.0638 WER, while four previously failing cases left the failure
list. Per-case latency was not supplied, so no latency improvement is claimed.

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
4. **Vocabulary sent through the wrong decoder channel.** Technical vocabulary now uses
   Faster-Whisper's `hotwords` option instead of pretending to be previous transcript,
   but only for English and automatic code-switching. A follow-up run exposed a
   regression where Latin hotwords pushed pinned Hindi toward fully romanized output;
   pinned Hindi therefore receives no Latin hotwords.
5. **Destructive post-processing.** Evaluation forced all Latin words in Hindi results
   through ITRANS without protecting enough legitimate developer vocabulary, while
   removing conversion entirely exposed outputs such as `Kaam ho gaya`. Hindi script
   normalization is retained, with a broader general developer vocabulary protected
   from conversion. Cleanup also retains deliberate double words while bounding runs
   of three or more.
6. **Mixed-language hotword over-bias.** The latest supplied run added failures across
   Hinglish variations (`*-03`/`*-04`), showing that an unconditional Latin vocabulary
   list can dominate acoustically ambiguous auto-language decoding too. Global hotwords
   are now limited to pinned English. Hinglish remains acoustically driven until a
   confidence-aware vocabulary selector is available.
7. **One-pass segment rejection.** Long continuous and repeated-word cases returned no
   transcription after otherwise useful segments crossed conservative no-speech,
   log-probability, or compression gates. A failed final now receives one prompt-free,
   hotword-free recovery decode with relaxed rejection gates. Corrupt/hallucinated text
   still passes through the existing low-quality rejection checks.

## Retest requirements

The environment used for this change does not provide the Windows neural fixture path
or Faster-Whisper model runtime, so no fabricated after-score is reported. On Windows,
rerun generation first so managed version-2 WAVs are replaced, then rerun the 19 cases,
all Hindi cases, related Hinglish cases, and finally the complete manifest. Preserve the
previous JSON and use `--baseline` to measure similarity, WER, latency, and regressions.
Any remaining failures must stay visible; do not change transcripts or thresholds.
