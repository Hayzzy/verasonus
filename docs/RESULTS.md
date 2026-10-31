# Results

All numbers come from the scripts named below and are saved under `reports/tables/` and
`reports/figures/`. Data: FakeMusicCaps (five generators: AudioLDM 2, MusicGen, MusicLDM,
Mustango, Stable Audio Open) as AI music and FMA small as real music, split by artist or
track so nothing leaks between train, val and test. Every clip is standardised the same way
(16 kHz mono, 7 kHz low-pass, matched loudness) and cut into 10-second windows with a
5-second hop. A clip's score is the mean of its window scores.

Sections 1–6 describe the first model (v1). The current model, v2, is in section 0.

## 0. Model v2 (current)

**What changed and why.** v1 learned FakeMusicCaps' research generators well but missed
real-world Suno and Udio songs, so v2 adds them:

- **More data.** 20,000 SONICS songs (Suno v2, v3, v3.5; Udio-30s and Udio-120s) join
  FakeMusicCaps and FMA small: 55,599 clips. Each SONICS song keeps the split SONICS gave it.
  Only Suno v3.5 and Udio-120s songs are in SONICS's train split, so Suno v2, Suno v3 and
  Udio-30s are versions the models never trained on.
- **Same encoding for every clip.** SONICS songs are low-bitrate MP3s, so every clip (and
  every upload) is passed through the same 32 kbps MP3 codec. Otherwise the bitrate alone
  would give the label away.
- **Long songs → one 20-second excerpt.** It starts 40% into the song, so full songs don't
  outnumber the short clips.
- **Edits during training.** Half the train clips (real and AI alike) were slowed, sped up
  (0.8–1.25×) and/or re-encoded as MP3 at 32–192 kbps.
- **Class weights.** These balance real against AI in the loss.
- **A third model.** The pretrained SONICS SpecTTTra (alpha, 120 s) is used without
  retraining.
- **Stems.** Vocals and instrumental are separated with Demucs and scored on their own. The
  new **partly AI** verdict means one stem scores as AI while the full mix does not.

**Held-out test clips** (`fit_ensemble.py` → `test_clip_models_vs_ensemble.csv`; 7,785
clips; threshold 0.5 after calibration). Flagged = share of clips called AI.

| Model | ROC-AUC | F1 | Real flagged | Suno v3.5 | Udio-120s | Suno v2* | Suno v3* | Udio-30s* |
|---|---|---|---|---|---|---|---|---|
| CNN | 0.999 | 0.978 | 0.2% | 100% | 99.9% | 99.4% | 100% | 76.7% |
| Gradient boosting | 0.930 | 0.927 | 21.1% | 100% | 99.5% | 95.6% | 97.8% | 74.1% |
| SpecTTTra (pretrained) | 0.741 | 0.617 | 8.8% | 100% | 100% | 13.5% | 97.6% | 3.5% |
| **Ensemble** (0.7 / 0.2 / 0.1) | **0.999** | **0.977** | **0.1%** | 100% | 99.9% | 99.4% | 100% | 75.5% |

\* versions in no train split. The FakeMusicCaps generators are flagged at 98.6–100% by the
ensemble. On its own, SpecTTTra fails on everything outside its two training generators. We
keep it in the ensemble at its minimum weight (0.1).

**Does singing get flagged?** Every SONICS song has vocals, so a model could learn "vocals =
AI". `evaluate.py --protocol stems --split test` (`false_alarms_by_vocals_test.csv`) puts each
test song in a group by the share of its energy that Demucs assigns to vocals:

| | Instrumental (<5%) | Some vocals (5–20%) | Vocal (>20%) |
|---|---|---|---|
| Real songs flagged (likely AI, ≥ 0.7) | 0.0% (n=174) | 0.0% (n=99) | 0.0% (n=127) |
| AI songs caught (≥ 0.7) | 96.7% (n=361) | 95.2% (n=126) | 94.6% (n=313) |

Real singing is not flagged, so the shortcut did not appear.

**Stems.** On val, each stem gets the lowest threshold that flags at most 5% of real songs:

- **Instrumental: 0.55.**
- **Vocals: 0.997.** Demucs-separated vocals of real songs often score high, probably because
  separation leaves its own artefacts. So the vocals threshold is strict, and only AI vocals
  scoring above 99.7% count. Hybrid songs built from the test set were checked by hand in
  the demo, not measured at scale (a bug in the hybrid step was fixed after the run).

**Edits** (`evaluate.py --protocol robustness`, 150 test clips per source,
`robustness_by_edit_v2a_edited_at_16k.csv`):

| Edit | ROC-AUC | AI caught | Real flagged |
|---|---|---|---|
| none | 0.999 | 97.0% | 0.0% |
| MP3 64–320 kbps | 0.999–1.000 | 96.3–97.7% | 0.0% |
| loudness −14 / −23 LUFS | 0.999 | 97.3% | 0.0% |
| speed 0.9, 1.1, 1.25 | 0.999–1.000 | 96.3–99.3% | 0.0% |
| speed 0.8 | 0.877 | 100% | **56.0%** |
| pitch ±1–2 semitones | 0.943–0.970 | 91.7–98.7% | **16.0–30.0%** |
| time stretch 0.9–1.1 | 0.968–0.978 | 94.7–96.3% | **13.3–15.3%** |
| noise 30 / 20 / 10 dB SNR | 0.991 / 0.972 / 0.886 | 87.7 / 83.3 / 62.7% | 0.7 / 0.0 / 2.7% |

**Main weakness:** real songs that are strongly slowed (0.8×), pitch-shifted or
time-stretched are often called AI. Part of this comes from the test itself. It edited audio
already reduced to 16 kHz, so a 0.8× slowdown empties the top of the band, which a real
"slowed" upload made from full-quality audio would not. The script now edits at twice the
rate, but that run was not finished. The other part comes from training: the models never
saw pitch shift or time stretch. The fix is to add both to the training edits and retrain.
Until then, **treat a "likely AI" verdict on slowed, sped-up or pitch-shifted remixes with
caution.**

**Generator ID** (`fit_generator_id.py --features handcrafted_v2`): 77.3% accuracy, macro F1
0.76 over seven platforms (Suno and Udio versions merged). In the leave-one-out test, the
"unknown" detector flags Stable Audio Open as unfamiliar 41% of the time, every other
generator only 1–15%, and 8–11% of known generators. It is not reliable.

## 1. Held-out tracks from seen generators

`scripts/fit_ensemble.py` → `test_clip_models_vs_ensemble.csv`, `roc_test_clips.png`,
`confusion_ensemble_test_clips.png`, `reliability_ensemble_test_clips.png`.
Test split, 3,610 clips, threshold 0.5.

| Model | ROC-AUC | F1 | EER | ECE | Real flagged as AI |
|---|---|---|---|---|---|
| Gradient boosting, 107 handcrafted features | 0.960 | 0.913 | 0.109 | 0.115 | 7.9% |
| CNN on log-mel spectrograms | 1.000 | 0.999 | 0.001 | 0.002 | 0.1% |
| Ensemble (0.8 CNN + 0.2 gradient boosting, calibrated) | 1.000 | 0.999 | 0.001 | 0.002 | 0.1% |

Share of each generator's test clips flagged as AI:

| Model | AudioLDM 2 | MusicGen | MusicLDM | Mustango | Stable Audio Open |
|---|---|---|---|---|---|
| Gradient boosting | 92% | 84% | 84% | 77% | 93% |
| CNN / ensemble | 100% | 100% | 99.6% | 100% | 99.5% |

Both models meet our aim (ROC-AUC ≥ 0.95, F1 ≥ 0.90) on seen generators.

**Ensemble.** Tuned freely on val, the weights went entirely to the CNN (val ROC-AUC 1.0). We
keep gradient boosting at a 0.2 weight, because its features give the explanation panel
something a person can read. On seen generators this costs nothing measurable. A
temperature of 0.34, fitted on val, turns the fused score into a probability.

**Thresholds.** `scripts/train.py` also picks a threshold on val that flags at most 5% of
real clips (`evaluation.max_fpr_real`). For the CNN that rule picks 0.0008, which flags 6.1%
of real test clips for no gain over 0.5 (0.1%). With a model this well separated, 0.5 is the
better choice. The app uses the calibrated probability with bands: ≥ 0.7 likely AI, ≤ 0.3
likely human, uncertain in between.

## 2. Generators never seen in training

`scripts/unseen_generators.py` (one model per held-out generator) and
`scripts/plot_unseen.py` → `unseen_generators_summary.csv`, `unseen_generators_caught.png`.
"Caught" is the share of the held-out generator's clips flagged as AI at the val-picked
threshold; about 5–6% of real clips are flagged in every fold.

| Held out | GB unseen ROC-AUC | GB caught | CNN unseen ROC-AUC | CNN caught |
|---|---|---|---|---|
| AudioLDM 2 | 0.911 | 65% | 1.000 | 100% |
| MusicGen | 0.634 | 26% | 0.852 | 54% |
| MusicLDM | 0.916 | 68% | 0.880 | 48% |
| Mustango | 0.728 | 32% | 1.000 | 100% |
| Stable Audio Open | 0.621 | 21% | 0.919 | 57% |

**This is the main finding.** Near-perfect scores on seen generators do not carry over to
new ones. Generators built like the training ones (AudioLDM 2 and MusicLDM are both latent
diffusion models) are still caught, but ones that work differently are often missed.

## 3. Which generator?

`scripts/fit_generator_id.py` → `generator_id_summary.json`,
`confusion_generator_id_test_clips.png`, `unknown_detector_leave_one_out.csv`,
`generator_map_pca.png`.

- **Generator ID** (gradient boosting on handcrafted features, AI clips only): 87.1%
  accuracy and 0.871 macro F1 on 2,761 test clips (chance is 20%).
- **"Unknown" detector** (IsolationForest on the known generators' features), tested
  leave-one-generator-out. It flags 5% of known generators' clips, as set. Of the held-out
  generator's clips it flags 26% for Stable Audio Open but only 2–5% for the other four. In
  practice it rarely says "unknown" when it should.
- **Clustering.** k-means on the AI test clips' features barely matches the generators
  (adjusted Rand index 0.10). The PCA map shows heavily overlapping generators. Telling
  generators apart needs supervision, at least with these features.

## 4. Explanations

- **Feature importance** (`gradient_boosting_permutation_importance.csv/.png`). The top
  features for gradient boosting are spectral contrast in the lowest band (mean), contrast
  variation in band 1, MFCC 1 and spectral flatness variation. Decoder-artefact periodicity
  ranks 8th.
- **Grad-CAM for the CNN** (`scripts/saliency_report.py` → `cnn_gradcam_profiles.png`,
  `cnn_gradcam_example.png`). On AI windows the CNN attends most to 4–7 kHz, just below the
  low-pass. This fits Afchar et al. (2024): neural decoders leave traces in the upper band.
  Attention also peaks about 0.4 s into every window, for real and AI alike. We have not
  explained this yet; because both classes show it, it cannot separate them on its own.
- **In the app**, each track's top features come from resetting one feature at a time to
  the median of real training windows and measuring how the score moves.

## 5. Shortcut checks and caveats

Because the CNN scores close to perfectly, we looked for shortcuts before trusting it:

- **Removed by design:** the band above 7 kHz (one feature there measured the resampler,
  not AI), loudness differences, and zero padding (only real clips had it, so spectrograms
  use each window's first 8 s).
- **Tested, no effect:** re-encoding AI test clips as MP3 (about 130 and 320 kbps) left 98%
  flagged. Shifting the window start by 1–2 s did not change scores.
- **Not ruled out:** style and recording differences between the sources. All real music is
  FMA (independent releases) and all AI music is FakeMusicCaps (prompted with MusicCaps
  captions). The CNN may partly learn "FMA versus MusicCaps style". The unseen-generator
  drop shows it learns generator-specific traces. Testing this needs real and AI music
  from the same source (e.g. SONICS).
- **Not done:** the robustness suite (codecs, pitch, time stretch, noise, phone
  re-recording). The Transformer, SpecTTTra and MERT/CLAP probes are not implemented.

## 6. Speed

The API analyses a 30-second clip in about 1.5 s on CPU (CNN + handcrafted features), after
a one-off model load.
