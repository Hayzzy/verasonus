<p align="center">
  <img src="web/public/icons/icon-192.png" width="88" alt="VeraSonus logo" />
</p>

<h1 align="center">VeraSonus</h1>

<p align="center">
  <b>Is this track AI-generated? Which parts? Why?</b><br />
  A pattern recognition system that detects, localises and explains AI-generated music.<br />
  <sub>NUS-ISS Pattern Recognition Systems practice module · Group 34 · Shrey Yadav &amp; Sushant Maramkulam Shiji</sub>
</p>

<p align="center">
  <a href="https://github.com/Hayzzy/verasonus/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/Hayzzy/verasonus/actions/workflows/ci.yml/badge.svg" /></a>
</p>

![VeraSonus web app, Phase 0 preview](docs/assets/web-preview.png)

## What it is

**VeraSonus** (Latin for "true sound") estimates how likely a music track is to be
AI-generated. It marks which segments look AI-generated, names the likely generator (or
"unknown"), and explains its decision so a person can check it. It should feel like a real
product, on par with tools such as IRCAM Amplify's AI music detector or SubmitHub's AI song
checker.

Our edge is **honesty and robustness**. Detectors that look near-perfect in the lab often break
on new generators and simple edits. So we measure how detection holds up on generators the
model has never seen and after everyday edits (MP3 compression, pitch shift, phone
re-recording), and we report those gaps as they are.

> **Status: working prototype (model v2).** Upload a track in the web app and get a verdict
> (likely AI, partly AI or likely human), an AI probability, separate scores for the vocals
> and the instrumental, a segment timeline, a generator guess and the features behind the
> score. It comes from an ensemble of our CNN, gradient boosting and the pretrained SONICS
> SpecTTTra, trained on Suno, Udio and five research generators. Results, including where the detector fails, are in
> [docs/RESULTS.md](docs/RESULTS.md). Parts still open are documented stubs
> (`NotImplementedError`); see the [roadmap](#roadmap).

## How the course requirements map to the code

The brief asks for at least three of four aspects. VeraSonus covers all four:

| Aspect | How VeraSonus covers it | Where (`src/verasonus/`) |
|---|---|---|
| **1. Supervised / unsupervised learning** | Real-vs-AI and which-generator classifiers; clustering and embedding maps of generators; outlier detection that flags unfamiliar generators as "unknown" | `models/classical.py`, `models/deep.py`, `models/probes.py`, `models/unsupervised.py` |
| **2. Machine / deep learning** | SVM, random forest and gradient boosting on handcrafted features; CNN and Transformer on spectrograms; MERT / CLAP embedding probes; SpecTTTra (SONICS) as reference baseline | `models/classical.py`, `models/deep.py`, `models/probes.py` |
| **3. Hybrid / ensemble approaches** | Stacking and weighted voting across model families; window scores fused into one calibrated track score and a timeline | `ensemble/fusion.py`, `ensemble/aggregation.py`, `ensemble/calibration.py` |
| **4. Intelligent sensing / sense-making** | Resampling and windowing; mel / CQT spectrograms, MFCCs, high-band and decoder-artefact features; Demucs stems; segment timeline, saliency maps, feature importance | `preprocessing/`, `features/`, `explain/`, `ensemble/aggregation.py` |

Final deliverables map to: the runnable system (`src/verasonus/api/` + `web/`), datasets
([docs/DATA.md](docs/DATA.md) + `scripts/`), report figures and tables (`reports/`), and code
and model files (this repo; model weights go to Hugging Face Hub or GitHub Releases, not git).

## Repository layout

```
verasonus/
├── src/verasonus/      Python package
│   ├── config.py         YAML config -> typed settings            (implemented)
│   ├── data/             dataset registry, clip table, splits      (implemented except download)
│   ├── preprocessing/    audio I/O, resampling, standardising, windowing (implemented)
│   ├── augmentation/     codecs, pitch/time/noise/loudness, phone playback, hybrid stems
│   ├── features/         spectrograms, handcrafted features, MERT/CLAP, Demucs
│   ├── models/           classical, deep, probes, unsupervised
│   ├── ensemble/         fusion, window -> track aggregation, calibration
│   ├── evaluation/       metrics, protocols, report outputs
│   ├── explain/          saliency maps, feature importance
│   └── api/              FastAPI service (/health works, /analyze returns 501 until Phase 3)
├── configs/            default.yaml, datasets.yaml, augment.yaml
├── scripts/            download, metadata, splits, train, evaluate, ONNX export, icons
├── notebooks/          notebooks (00_colab_setup.ipynb is the optional Colab fallback)
├── tests/              pytest, synthetic audio only (no downloads)
├── web/                React + Vite + TypeScript + Tailwind PWA
├── extension/          Chrome extension, Manifest V3 skeleton (Phase 4)
├── docs/               DATA.md, AI_USAGE.md
├── reports/            figures/ and tables/ for the report
└── data/               datasets (ignored by git)
```

## Setup

### Local (Windows, macOS, Linux)

You need Python 3.12+ and Node.js 22.12+ (we use 24 LTS). Training also needs an NVIDIA GPU
and ffmpeg; see [Training on your own PC](#training-on-your-own-pc-gpu).

```bash
git clone https://github.com/Hayzzy/verasonus.git
cd verasonus
python -m venv .venv                # Windows: py -3.14 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements/dev.txt # exact pinned versions
pip install -e . --no-deps          # the package itself, editable
pytest                              # tests
ruff check . && ruff format --check .
```

**Web app** (Phase 0 preview: upload page with a local waveform preview)

```bash
cd web
npm ci
npm run dev      # http://localhost:5173 (proxies /api to the FastAPI service)
npm run build    # type-check + production build
```

**API**

```bash
uvicorn verasonus.api.main:app --reload   # docs at http://127.0.0.1:8000/docs
```

**iPhone.** The web app is a PWA. Once it is deployed over HTTPS (Phase 3), open it in Safari
and choose Share → Add to Home Screen. It then runs full screen and can record from the
microphone.

### Training on your own PC (GPU)

We train on our own machine with an NVIDIA GPU.

1. **Put the data on a roomy drive** and point the project at it, e.g. in `.env`:
   `VERASONUS_DATA_ROOT=D:/verasonus-data`
2. **Install PyTorch with CUDA, then the rest of the deep learning stack.** It is heavy, so
   agree as a team first. RTX 50-series GPUs need a CUDA 12.8+ build:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu130
   pip install --no-deps torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
   pip install -r requirements/ml.txt
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
3. **Install ffmpeg** (needed for the codec edits from Phase 1) and check `ffmpeg -version`.

Use mixed precision and modest batch sizes, and save checkpoints often so a long run can
resume. Generators that need more GPU memory than you have (such as YuE) can run on a rented
cloud GPU.

### Google Colab (optional fallback)

Training runs locally by default (see above). Colab stays available as a fallback, for example
for a teammate without an NVIDIA GPU. The code is the same either way; only
`VERASONUS_DATA_ROOT` and the torch install differ.

1. In Colab: File → Open notebook → GitHub. Tick "Include private repos" and open
   `notebooks/00_colab_setup.ipynb`.
2. Add a Colab secret `GH_TOKEN`: a GitHub fine-grained token with read access to this repo.
3. Run all. The notebook:
   - clones the repo and installs it with `pip install -e ".[api,dev]"` (version ranges, so
     Colab keeps its preinstalled torch)
   - mounts Google Drive and points `VERASONUS_DATA_ROOT` at it
   - runs the tests

### Configuration

| File | What it controls |
|---|---|
| `configs/default.yaml` | Paths, target sample rate (16 kHz), window length/hop, split ratios, held-out generators, seed |
| `configs/datasets.yaml` | Data sources: role, access, licence, links |
| `configs/augment.yaml` | Robustness suite: codecs and bitrates, pitch, stretch, noise, loudness, phone playback |

Override in code with `load_config(overrides={"audio.target_sample_rate": 32000})`.
Environment variables (see `.env.example`): `VERASONUS_DATA_ROOT` (where data lives) and
`HF_TOKEN` (gated Hugging Face models).

## Data

Audio is **never** committed; see [docs/DATA.md](docs/DATA.md) for sources, licences and
rules.

1. `python scripts/download_data.py --list` shows every source with its role and licence.
2. Download each source into `data/raw/<source>/`. The download script is a Phase 1 stub;
   follow the access notes in DATA.md until then.
3. `python scripts/build_metadata.py --sources fma fakemusiccaps` writes `metadata.csv` in
   the data root, one row per clip. Scanners exist for these two sources so far.
4. `python scripts/make_splits.py`, or `--holdout musicgen` for unseen-generator tests, adds a
   split column. Rows with the same `group_id` (an FMA artist, otherwise a track) stay
   together.
5. `python scripts/standardise_audio.py` writes a 16 kHz mono FLAC copy of every clip to
   `interim/`, processed the same way for real and AI audio (low-passed at 7 kHz and
   matched in loudness, because both separated the classes on their own). Add `--audit 200` to compare
   bandwidth per generator. Rerunning skips clips that are already done.
6. `python scripts/extract_features.py` cuts every clip into windows and saves one feature
   vector per window to `processed/handcrafted/` (`--limit 200` for a quick trial).
   `--features mel` saves a log-mel spectrogram of each window to `processed/mel/` instead
   (about 7 GB), the input for the CNN.
7. `python scripts/train.py --model gradient_boosting` (or `--model cnn`, on the GPU) trains
   on train, tunes on val and scores test, and
   `python scripts/unseen_generators.py` reruns it once per AI generator with that generator
   held out, to see how well it catches a generator it never saw. Both call a clip AI above a
   threshold picked on val so that at most 5% of real clips are flagged
   (`evaluation.max_fpr_real`).
8. `python scripts/fit_ensemble.py` fuses the CNN and gradient boosting, calibrates the fused
   score and compares the three on test. `python scripts/fit_generator_id.py` trains the
   generator classifier and the "unknown" detector, maps how generators cluster and ranks
   the handcrafted features. Both save what the API needs to `checkpoints/` and their
   tables and figures to `reports/`.

## Run the prototype

After steps 1–8, or with the trained model files from the private
[`v2-models` release](https://github.com/Hayzzy/verasonus/releases/tag/v2-models) (`cnn.pt`,
`gradient_boosting.joblib`, `ensemble.joblib`, `generator_id.joblib`) saved in
`checkpoints/`, e.g. `gh release download v2-models --dir checkpoints`. The first run also
downloads the pretrained SpecTTTra and Demucs weights. Needs `pip install -r
requirements/ml.txt`, plus `sonics` and `demucs`. `sonics` also needs `torchaudio`, and
`timm` needs `torchvision`: install both from the PyTorch index as in step 2 of
[Training on your own PC](#training-on-your-own-pc-gpu), or pip fetches CPU-only builds:

```bash
uvicorn verasonus.api.main:app --port 8000   # needs torch (CPU is enough)
cd web && npm run dev                         # http://localhost:5173
```

Drop an MP3, WAV, FLAC or OGG file of at least 8 seconds on the upload card and press
Analyse. The API standardises the audio exactly like the training clips, scores 10-second
windows with all three models, separates and scores the vocals and instrumental, and deletes
the upload when it is done.

## Evaluation

We report the following, and never inflate it:

- **Metrics:** accuracy, precision, recall, F1, ROC-AUC, equal error rate, calibration (ECE)
  and confusion matrices. The false-positive rate on real music is reported separately.
- **Protocols:**
  - held-out tracks from seen generators
  - leave-one-generator-out
  - the robustness suite

  Each comes with per-generator and per-edit reports.
- **Aims:** ROC-AUC ≥ 0.95 and F1 ≥ 0.90 on held-out tracks from seen generators. The
  unseen-generator and robustness gaps are measured, then reduced as far as we can.
- **Sanity rule:** suspiciously perfect accuracy means we look for a shortcut before
  celebrating.
- **App performance:** a 30-second clip is analysed in a few seconds on CPU, using ONNX where
  it helps.

## Roadmap

- [x] **Phase 0:** repository scaffold, configs, tested utilities, web and extension skeletons, CI, docs
- [x] **Phase 1:** data pipeline, metadata table, splits, classical baselines, SONICS, the
  MP3 robustness edits (open: AAC/Opus and phone re-recording, which need ffmpeg)
- [x] **Phase 2:** CNN, ensemble with the pretrained SpecTTTra, calibration, Demucs vocals and
  instrumental scores, generator ID, clustering, "unknown" detector, feature importance,
  Grad-CAM, evaluation reports (open: training with pitch and tempo edits, Transformer,
  MERT/CLAP probes)
- [x] **Phase 3:** FastAPI `/analyze` and the web app showing its results, including the
  "partly AI" verdict (open: microphone recording, history, share card, hosting)
- [ ] **Phase 4:** Chrome extension (tab capture), then iPhone options
- [ ] **Throughout:** report-ready figures and tables saved to `reports/`

## Design

The web app, extension and icons use the theme of our slides:

- **Colours:** black background; flame `#FF7A00` and ember `#D56A46` gradients; apricot
  `#F7A760` labels; thin ember outlines on rounded cards.
- **Fonts:** Inter headings and Source Sans 3 body text.

The design tokens live in [`web/src/index.css`](web/src/index.css); the icons come from
`scripts/make_icons.py`.

## Team, AI usage and licence

- Shrey Yadav and Sushant Maramkulam Shiji.
- AI assistance is logged in [docs/AI_USAGE.md](docs/AI_USAGE.md), as the course requires.
- Code is released under the [MIT licence](LICENSE). Datasets keep their own licences.

## Key references

- Afchar et al. (2024). *Detecting music deepfakes is easy but actually hard.*
- Rahman et al. (ICLR 2025). *SONICS: Synthetic Or Not – Identifying Counterfeit Songs.*
- Comanducci et al. (2025). *FakeMusicCaps: a dataset for detection and attribution of synthetic music.*
- Batra et al. (TMLR). *Melody or Machine: detecting synthetic music with dual-stream contrastive learning.*
- Rigaud et al. (2026). AI-generated stems in hybrid music (Deezer).
