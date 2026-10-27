# CLAUDE.md: VeraSonus

Context for Claude Code sessions in this repository. Read it before changing anything.

## Project

VeraSonus ("true sound" in Latin) is the NUS-ISS Pattern Recognition Systems (PRS)
practice-module project of Group 34: Shrey Yadav and Sushant Maramkulam Shiji.

**What it does.** It estimates whether a music track is AI-generated, shows which parts look
AI-generated, names the likely generator (or "unknown"), and explains its decision.

**Bar to meet.** It must feel like a real product, on par with IRCAM Amplify's AI music
detector and SubmitHub's AI song checker.

**Our edge.** Honesty and robustness: we measure detection on unseen generators and everyday
audio edits.

## Course requirements

Deliverables:
- a runnable system
- datasets
- a final report (tools and techniques, system design and models, performance, findings and
  discussion)
- code and model files
- a 10–15 minute video
- slides from two presentations

The project must show at least three of the four aspects. We cover all four, so keep each one
visible in the code and the README:

1. **Supervised / unsupervised learning:** `models/classical.py`, `models/deep.py`,
   `models/probes.py`, plus `models/unsupervised.py` (clusters, embedding maps, "unknown"
   outlier detection)
2. **Machine / deep learning:** `models/classical.py` (SVM, random forest, gradient boosting),
   `models/deep.py` (CNN, Transformer, SpecTTTra baseline), `models/probes.py` (MERT/CLAP)
3. **Hybrid / ensemble:** `ensemble/fusion.py`, `ensemble/aggregation.py`,
   `ensemble/calibration.py`
4. **Intelligent sensing / sense-making:** `preprocessing/`, `features/`, `explain/`, and the
   timeline in `ensemble/aggregation.py`

**Where training runs.** On Shrey's laptop: an RTX 5070 Laptop GPU (8 GB VRAM), 32 GB RAM
and a 24-core CPU, with datasets on a separate SSD via `VERASONUS_DATA_ROOT`.
- Colab is an optional fallback (`notebooks/00_colab_setup.ipynb`). Keep code free of
  machine-specific paths and pick the device at runtime (`cuda` if available, else CPU).
- Design within 8 GB of VRAM: mixed precision and modest batch sizes.
- Generators: MusicGen small/medium and Stable Audio Open in half precision fit; ACE-Step
  needs CPU offloading. YuE (~24 GB) and MusicGen-large need a rented GPU.

## Phases

- **Phase 0, done 2026-10-27:** scaffold.
  - Implemented and tested: `config.py`, `preprocessing/audio_io.py`,
    `preprocessing/windowing.py`, `data/splits.py`, and `data/sources.load_registry`.
  - Everything else is a stub: a docstring and signature that raises
    `NotImplementedError("Phase N")`.
- **Phase 1 (in progress):** data pipeline, metadata table, splits, augmentation suite,
  classical baselines.
  - Done: FakeMusicCaps, FMA small and FMA metadata are in the data root, and the clip table
    covers them (`data/sources.scan`, `data/metadata.build_metadata` and
    `validate_metadata`, written to `<data_root>/metadata.csv`).
- **Phase 2:** deep models, embedding probes, ensemble, calibration, full evaluation reports.
- **Phase 3:** FastAPI service and the polished PWA web app (the working prototype).
- **Phase 4:** Chrome extension (MV3, `chrome.tabCapture` + offscreen document), then iPhone
  options. The PWA is the free iPhone route; a native SwiftUI app only if the team decides it
  is worth it.
- **Throughout:** report-ready figures and tables go to `reports/`.

## Commands

- Python (3.12+): `pip install -r requirements/dev.txt && pip install -e . --no-deps`
- GPU stack (heavy, ask first): `pip install torch --index-url https://download.pytorch.org/whl/cu130`,
  then `pip install -r requirements/ml.txt`
- Test: `pytest`. Lint: `ruff check . && ruff format --check .`
- API: `uvicorn verasonus.api.main:app --reload`
- Web (`web/`): `npm ci`, `npm run dev` (port 5173; proxies `/api` to port 8000), and
  `npm run build` (tsc + vite)
- Extension (`extension/`): `npm ci`, then `npm run build` (tsc to `dist/`)
- Icons: `python scripts/make_icons.py` regenerates the PWA and extension icons

## Conventions

- Explain every file you create or change in plain language. The team must account for it in
  the report.
- Change only what is needed. Don't rewrite files nobody asked you to touch.
- Ask before installing heavy packages locally (torch, transformers, demucs). Listing them in
  requirements is fine.
- Keep it simple enough for a two-person student team to maintain.
- The team implements the ML logic themselves. Don't fill in stubs unless asked.
- Keep settings in `configs/*.yaml` (paths, sample rate, windows, splits, augmentations) and
  load them with `verasonus.config.load_config`. Never hard-code them.
- Import torch and transformers inside functions or under `TYPE_CHECKING`, so the package
  imports without them.
- Exact pins go in `requirements/*.txt` (local and CI); version ranges go in `pyproject.toml`
  (so a Colab fallback keeps its preinstalled packages). torch is never in
  `requirements/*.txt`: install its CUDA build from the PyTorch index (cu130 for RTX 50-series).
- Add an entry to `docs/AI_USAGE.md` for every AI-assisted session (a course requirement).
- Never commit audio, datasets, checkpoints, `.env` or secrets. Commit or push only when asked.

## Data rules

- **Metadata table** (`data/metadata.py`): one row per clip with track and group ids, source
  dataset, label (real/ai/partly_ai), generator, a path relative to the data root, original
  sample rate, codec, duration, licence, split and genre.
- **Same processing for every clip:** resample everything to one configurable rate (default
  16 kHz) and encode real and AI audio the same way. Otherwise models learn format or
  bandwidth; MusicGen is 32 kHz, so it has nothing above 16 kHz.
- **Splits:** rows with the same `group_id` stay in one split (all windows of a track, all
  tracks of an FMA artist). FakeMusicCaps clips are their own groups, because every caption
  exists for every generator. Leave-one-generator-out puts whole generators in
  `test_unseen`. Balance genres where possible.
- **Robustness suite** (`configs/augment.yaml`): MP3/AAC/Opus at several bitrates, ±2
  semitone pitch shift, time stretch, added noise, loudness normalisation, and simulated phone
  playback and re-recording.
- **Sources:** see `docs/DATA.md` and `configs/datasets.yaml`.
  - SONICS: AI MP3s download directly; real songs are YouTube IDs only.
  - FakeMusicCaps: Zenodo, CC BY-NC.
  - Melody or Machine: controlled access.
  - FMA and MTG-Jamendo: shareable real music.
  - MUSDB18-HQ: stems for hybrid tracks via EnCodec (Rigaud et al. 2026).
  - Own set: MusicGen, Stable Audio Open and ACE-Step; YuE only if GPU time allows.

## Evaluation rules

- **Metrics:** accuracy, precision, recall, F1, ROC-AUC, EER, ECE and confusion matrices. The
  false-positive rate on real music is reported separately.
- **Protocols:**
  - held-out tracks from seen generators
  - leave-one-generator-out
  - the robustness suite

  Report per generator and per edit.
- **Aims:** ROC-AUC ≥ 0.95 and F1 ≥ 0.90 on held-out tracks from seen generators. Measure the
  unseen-generator and robustness gaps and reduce them. Report honestly; never inflate.
- **Suspiciously perfect accuracy:** assume a shortcut and investigate before celebrating
  (`evaluation/protocols.shortcut_checks`).
- **App performance:** a 30-second clip is analysed in a few seconds on CPU (ONNX where it
  helps).

## Product

- **API:** FastAPI.
  - `POST /analyze` returns the AI-likelihood, confidence, segment timeline, likely generator
    or "unknown", and explanation data.
  - Audio is deleted after analysis by default.
  - The schemas in `api/schemas.py` are mirrored in `web/src/lib/api.ts`; keep them in sync.
- **Web:** React 19, Vite, TypeScript and Tailwind v4.
  - A PWA: installs on iPhone via Safari "Add to Home Screen" and records from the microphone.
  - Planned features: drag-and-drop and microphone input; a verdict with a calibrated score;
    a waveform timeline; an explanation panel; history; a shareable result card.
- **Theme** (from the slides, `VeraSonus - Group34.pptx`):
  - Background `#000`.
  - Cards `#0A0A0A`: rounded, with a 1.5px `#D56A46` outline. Warm cards are a `#30180F` →
    `#20100A` gradient.
  - Flame `#FF7A00` → ember `#D56A46` gradients for big numbers and primary buttons.
  - Apricot `#F7A760` for uppercase, letter-spaced labels; ash `#6A6A6A` lines; mist
    `#B3B3B3` secondary text.
  - Inter bold headings (−0.04em tracking) and Source Sans 3 body text.
  - Lucide icons inside outlined circles, soft radial glows, and orange-to-pink acrylic panels
    drawn in CSS (never the deck's stock photo).
  - Tokens live in `web/src/index.css`.
- **Deployment:** stay free. Run locally for demos; use a free tier (e.g. Hugging Face Spaces)
  for the API.

## References

- Afchar et al. 2024, "Detecting music deepfakes is easy but actually hard"
- SONICS (Rahman et al., ICLR 2025)
- FakeMusicCaps (Comanducci et al., 2025)
- Melody or Machine (Batra et al., TMLR)
- Rigaud et al. 2026, AI-generated stems in hybrid music (Deezer)

## GitHub

Private repository `Hayzzy/verasonus` under the MIT licence. Sushant (`MSSushant`) is a
collaborator with write access. Real co-ownership would need a GitHub organisation.
