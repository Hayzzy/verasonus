# AI usage log

Our course requires us to document AI assistance. We used an AI coding assistant (Claude Code)
for parts of this project. Each entry says what we asked for, what it produced and how we
checked it.

| Date | Asked for | What the assistant produced | How we checked it |
|---|---|---|---|
| 2026-10-27 | Repository setup | Structure (approved by us), configs, tested utilities (config, audio I/O, windowing, splits, dataset registry), stubs for everything else, web app and extension skeletons, CI, docs | Tests, lint and builds pass; web app checked in a browser |
| 2026-10-27 | Local training instead of Colab | Hardware and PyTorch compatibility check; docs updated for local training (Colab kept as fallback) | Read through the changes |
| 2026-10-27 | First datasets | Downloaded FakeMusicCaps, FMA small and FMA metadata; found 6 FMA clips that cannot be decoded | Checksums match the published ones; every clip decoded |
| 2026-10-27 | Clip table | `scan`, `build_metadata`, `validate_metadata`; a `group_id` so no FMA artist spans two splits | 14 new tests; table of 35,599 clips validated |
| 2026-10-30 | Splits and standardised audio | Ran the splits; `standardise_file`, `effective_bandwidth_hz` and `scripts/standardise_audio.py` (16 kHz mono FLAC copies of every clip) | 4 new tests; all 35,599 clips converted; bandwidth similar across real and AI clips |
| 2026-10-30 | Window fix and feature caching | Raised `min_last_seconds` to 8 so no window is mostly padding; `features/cache.py` and `scripts/extract_features.py` (runs the feature functions on every window and saves the results) | Window counts checked on the real clip table; 5 new tests; trial on 400 real clips with a stand-in feature |
| 2026-10-30 | First classical baseline | Handcrafted features, classical models (SVM, random forest, gradient boosting), metrics and `scripts/train.py` | 9 new tests; trained and evaluated on our data |
| 2026-10-30 | Check the near-perfect baseline for shortcuts | Found that one feature in the 6.4–8 kHz band measured the resampler, not AI (controlled re-resampling test); added a 7 kHz low-pass and loudness matching to standardisation | 3 new tests; low-pass comparison on a 12% sample; full rebuild and retrain |
| 2026-10-31 | Unseen-generator test | `scripts/unseen_generators.py` (holds each generator out in turn, retrains, scores it) | Ran all five folds; real test clips identical across folds; seen scores match the main baseline |

Decisions were ours throughout: the structure, the private repo and MIT licence, which
datasets to use, and training locally.
