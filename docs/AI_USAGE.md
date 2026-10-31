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
| 2026-10-31 | Decision threshold | `threshold_at_fpr` and `evaluation.max_fpr_real`: the AI threshold is picked on val clips so at most 5% of real val clips are flagged; both training scripts use it | 1 new test; retrained both models and reran the unseen test; 0.5 results kept alongside |
| 2026-10-31 | CNN | Log-mel spectrogram cache (`--features mel`), the CNN and its training loop in `models/deep.py`, CNN support in `train.py` and `unseen_generators.py` | 7 new tests; full training and all five unseen-generator folds run; MP3 re-encoding and crop-offset checks for shortcuts (neither changed the scores) |
| 2026-10-31 | Finish a working prototype | Ensemble, calibration and timeline (`ensemble/*`, `scripts/fit_ensemble.py`); generator ID, "unknown" detector, clustering and feature importance (`models/unsupervised.py`, `explain/importance.py`, `scripts/fit_generator_id.py`); report figures (`evaluation/reports.py`, `scripts/plot_unseen.py`); `POST /analyze` (`api/inference.py`) and the web app's result view; `docs/RESULTS.md` | 9 new tests; every script run on our data; real and AI test clips analysed through the web app in a browser |

| 2026-10-31 | Model v2: Suno, Udio and partly-AI tracks | SONICS scanner and `--append` in `build_metadata.py`; shared MP3 codec and long-song excerpts (`preprocessing/standardise.py`); MP3 and speed edits (`augmentation/*`) and seeded training edits (`features/cache.py`); class weights; pretrained SpecTTTra (`models/deep.py`, `scripts/eval_spectttra.py`); Demucs stems (`features/separation.py`); 3-model ensemble; `partly_ai` verdict and stem scores in the API and web app; `scripts/evaluate.py` (stems, vocals, robustness); Grad-CAM | 3 new test files; all of v2 retrained and evaluated on held-out SONICS, FakeMusicCaps and FMA test clips; demo tracks (including a built hybrid) checked through the API and the web app; a robustness weakness (slowed and pitch-shifted real songs) found and reported, not hidden |
| 2026-10-31 | Set up the prototype on Sushant's laptop | Synced the repo with GitHub, downloaded the `v2-models` release, found that `sonics` also needs `torchaudio` (not listed anywhere) and that `timm` pulled a CPU-only `torchvision`; added the PyTorch-index install line to `requirements/ml.txt` and the README | 93 tests pass; a 15-second test clip analysed through the API (HTTP 200, full result) |

Decisions were ours throughout: the structure, the private repo and MIT licence, which
datasets to use, and training locally.
