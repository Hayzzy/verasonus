# Data

> **Rule: audio is never committed to git.** Not raw downloads, not generated clips, not
> edited copies. `.gitignore` blocks audio formats and everything under `data/` except its
> README. Datasets live on local disks or Google Drive; the repository only holds download
> scripts, loaders and the metadata schema.

Licences and access were checked on 2026-10-27. Re-check each dataset card before you
download, and keep all use academic and non-commercial.

## Sources

| Source | What it gives us | Role | Access | Licence |
|---|---|---|---|---|
| [SONICS](https://huggingface.co/datasets/awsaf49/sonics) (ICLR 2025) | ~97K full songs: 49,074 AI (Suno, Udio), 48,090 real | Main training set | AI songs: direct MP3 download. Real songs: YouTube IDs only | CC BY-NC 4.0 (code/models MIT) |
| [FakeMusicCaps](https://zenodo.org/records/15063698) | 27,605 ten-second clips from MusicGen, AudioLDM2, MusicLDM, Mustango, Stable Audio Open | Which-generator attribution | Open Zenodo download (12.9 GB) | CC BY-NC 4.0 |
| [Melody or Machine](https://arxiv.org/abs/2512.00621) (TMLR) | ~130K songs incl. Suno v4, Riffusion, YuE, DiffRhythm and partly-AI tracks | Unseen-generator testing | Controlled access for verified researchers; real songs are YouTube links | CC BY-NC 4.0 (AI songs) |
| [FMA](https://github.com/mdeff/fma) | Creative Commons music across 161 genres | Extra real music, shareable | Direct zip downloads | Per-track artist licences (mostly CC); metadata CC BY 4.0 |
| [MTG-Jamendo](https://github.com/MTG/mtg-jamendo-dataset) | 55,701 Creative Commons tracks | Extra real music, shareable | Download script in the MTG repo | Per-track CC; metadata CC BY-NC-SA 4.0; non-commercial research only |
| [MUSDB18-HQ](https://zenodo.org/records/3338373) | 150 songs with vocals / drums / bass / other stems (44.1 kHz WAV) | Hybrid human + AI tracks | Open Zenodo download (22.7 GB) | Non-commercial, educational use |
| Our own set | Clips we generate on our own GPU, plus edited copies | Fresh test set from generators we control | Our data drive | Each model's terms (below) |

The same facts are machine-readable in [`configs/datasets.yaml`](../configs/datasets.yaml)
(`python scripts/download_data.py --list`).

### Notes per source

- **SONICS.** In use: all 49,074 AI songs are downloaded (10 zips, 32.6 GB, into
  `raw/sonics/fake_songs/part_XX/`, with `fake_songs.csv` beside them). Model v2 trains on
  parts 1–4 (20,000 songs). Each song keeps SONICS's own split, so the pretrained SONICS
  models are never tested on their own training songs. The songs are low-bitrate MP3s
  (about 37 kbps), so every clip in our table is re-encoded with one shared 32 kbps MP3
  codec (`audio.shared_codec`). Songs over 60 s are cut to a 20 s excerpt starting 40% in.
  The AI songs are MP3 files (`fake_songs/*.mp3`) on Hugging Face and Kaggle.
  Real songs are only listed as YouTube IDs in `real_songs.csv`. Downloading YouTube audio may
  break YouTube's terms, so check with the module team first. Otherwise, use FMA or
  MTG-Jamendo as the real side and match genres where possible.
- **FakeMusicCaps.** Every clip is labelled by generator, which makes it our attribution set.
  The matching real MusicCaps clips are YouTube-only. Clips are 16 kHz mono float WAV,
  10.0–10.24 s long. All five generators used the same 5,521 MusicCaps captions, so each file
  name appears once per generator. The clip table treats each clip as its own group: grouping a
  caption's five clips would leave leave-one-generator-out nothing to split. The cost is that
  one caption can appear in both train and test.
- **Melody or Machine.** It trains on Suno v2/v3.5, Udio v1.5 and DiffRhythm. Its
  out-of-distribution test set uses Riffusion, Suno v3, Suno v4, YuE and voice clones. Its
  "Mostly Fake" tier has partly-AI tracks: real lyrics over AI audio, and voice-cloned covers.
  Request access from the authors early.
- **FMA and MTG-Jamendo.** Openly licensed real music, and the shareable part of our dataset.
  Respect each track's NC/ND/SA terms and keep attribution. For MTG-Jamendo, `--type audio-low`
  saves space (~156 GB for `raw_30s`). For FMA, also download `fma_metadata.zip`: its
  `tracks.csv` gives each clip's genre, artist and licence. Six fma_small clips cannot be
  decoded, so `exclude` in `configs/datasets.yaml` skips them. `fma_medium.zip` (25,000
  clips, 22 GiB) is downloaded but not yet in the clip table. It would add more real music,
  including more songs with vocals. Where a track is in both, fma_small's copy wins.
- **MUSDB18-HQ.** Following Deezer's 2026 work (Rigaud et al.), we pass chosen stems through
  a neural audio codec such as EnCodec and remix them. That gives hybrid tracks where we know
  exactly which parts are "AI".
- **Our own set.** Generated on our own GPU:
  - MusicGen small/medium (weights CC BY-NC 4.0)
  - Stable Audio Open in half precision (Stability AI Community License; gated, so accept its
    terms on Hugging Face and set `HF_TOKEN`)
  - ACE-Step with CPU offloading (Apache 2.0)
  - YuE (Apache 2.0), only on a rented GPU (it needs about 24 GB of VRAM)

  Edited copies come from the robustness suite.

## Layout on disk

```
<data_root>/             ./data, or VERASONUS_DATA_ROOT (e.g. D:/verasonus-data)
├── raw/<source>/        downloads, untouched
├── interim/<source>/    standardised audio: mono, target sample rate, FLAC / PCM_16
│                        (scripts/standardise_audio.py; mirrors raw/, so raw/x/y.mp3 -> interim/x/y.flac)
├── processed/           features, embeddings and caches
│                        (processed/<name>/: features.npy, windows.csv, columns.txt)
└── metadata.csv         one row per clip
```

## Metadata table

One row per clip ([`src/verasonus/data/metadata.py`](../src/verasonus/data/metadata.py)):
`clip_id`, `track_id`, `group_id`, `source_dataset`, `label` (real / ai / partly_ai),
`generator`, `path` (relative to the data root), `original_sample_rate`, `codec`,
`duration_seconds`, `licence`, `split`, `genre`.

`group_id` is what a split keeps together: the artist for FMA, otherwise the track. Build the
table with `python scripts/build_metadata.py --sources fma fakemusiccaps` (the sources with a
scanner so far). It checks every row before writing.

## Rules that keep results honest

1. **Process real and AI audio the same way.** Decode, down-mix, resample to
   `audio.target_sample_rate` (16 kHz by default), store as FLAC / PCM_16, and optionally
   pass everything through one shared lossy codec. Otherwise models learn bandwidth or file
   format instead of "AI": MusicGen outputs 32 kHz audio (nothing above 16 kHz), while many
   real tracks are 44.1 kHz. We found this in practice: a first baseline scored ROC-AUC 0.9997
   almost entirely from the 6.4–8 kHz band, where the resampler used on the real clips leaves
   a steep cut-off. Standardisation now low-passes every clip at 7 kHz (`audio.lowpass_hz`)
   and matches loudness (`audio.target_rms_dbfs`).
2. **Split by group.** All windows of a track share one split, and so do all tracks of an FMA
   artist (`group_id`, [`data/splits.py`](../src/verasonus/data/splits.py)).
   Leave-one-generator-out moves whole generators into `test_unseen`.
3. **Balance genres** where possible: add `genre` to `splits.stratify_columns`.
4. **Audit shortcuts.** Duration, sample rate, codec, loudness, leading silence and
   bandwidth should not predict the label on their own (`evaluation.protocols.shortcut_checks`).
5. **Keep evaluation edits apart.** Robustness copies of test audio are for evaluation only;
   training augmentation uses training audio only.

## Sharing

Only openly licensed material may be shared outside the team: FMA and MTG-Jamendo tracks whose
licences allow it, and our own generated clips where the model licence allows. When in doubt,
share the metadata table and download scripts instead of audio. If a rights holder asks us to
remove something, delete it from our storage and note it in the metadata table.
