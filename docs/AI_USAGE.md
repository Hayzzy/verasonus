# AI usage log

Our course requires us to document AI assistance. Add an entry for every session in which an
AI tool helped. Keep each entry factual: what was asked, what the AI produced, what we checked
or changed, and what is still ours to do.

| Date | Tool | Prompted by | Summary |
|---|---|---|---|
| 2026-10-27 | Claude Code (Claude Opus 5.5) | Shrey Yadav | Phase 0 repository scaffold |

---

## 2026-10-27 · Phase 0 repository scaffold

**Tool.** Claude Code desktop app, model Claude Opus 5.5.

**Prompted by.** Shrey Yadav. Inputs were the team's project brief and the slides from the
first presentation (`VeraSonus - Group34.pptx`).

**What we asked for.**
- Propose a repository structure and wait for approval.
- Python packaging with pinned requirements, and YAML configs.
- Module stubs, plus small tested utilities.
- A React + Tailwind web app and a Manifest V3 extension skeleton, themed like the slides.
- pytest tests and GitHub Actions CI.
- README, `docs/DATA.md`, `CLAUDE.md` and this log.
- A first commit, a private GitHub repo, and Sushant (`MSSushant`) as collaborator.

**Decisions the team made.**
- Approved the proposed structure.
- Repo lives at `Hayzzy/verasonus` on Shrey's account, with Sushant as collaborator. GitHub
  allows two owners only through an organisation.
- Private repository, MIT licence.
- Node.js installed as a portable copy in Shrey's user folder, not system-wide.

**What the AI did.**
- **Theme:** read the slide deck (colours, fonts, layout motifs) to build the web theme, and
  recreated the title-slide look in CSS instead of reusing the deck's stock photo.
- **Online checks:** current package versions, and dataset facts (licences, access, sizes).
  Pins use releases that were at least a week old where possible.
- **Code:** wrote every file in the first commit.
  - Implemented: config loading, audio loading and resampling, windowing, track-level and
    generator-held-out splits, and the dataset registry.
  - Everything else (features, models, ensemble, evaluation, augmentation, API inference) is
    documented stubs for the team to implement.
- **Tests and CI:** 32 pytest tests using synthetic audio only, plus a GitHub Actions workflow
  (ruff, pytest, web build, extension build).
- **Icons and visuals:** the logo icons (`scripts/make_icons.py`) and the README screenshot.

**How it was checked.**
- ruff and all 32 tests pass locally (Python 3.14).
- The web app type-checks and builds, and was inspected in a browser at desktop and phone
  widths, including the file-upload preview.
- The extension compiles.

**Follow-up the same day: local training instead of Colab.** Shrey asked to train on a personal
PC instead of Colab.
- Claude checked the hardware: RTX 5070 Laptop GPU (8 GB VRAM), 32 GB RAM, a 24-core CPU and a
  free 1 TB SSD for data.
- It confirmed that PyTorch's CUDA 13.0 build supports this GPU on Windows with Python 3.14,
  and that the rest of the deep learning stack has matching wheels.
- It updated the docs, configs and docstrings to a local-first workflow. The Colab notebook
  stays as an optional fallback.
- Nothing heavy was installed without the team's approval.

**Follow-up the same day: first datasets.** Shrey asked Claude to set up the data drive and
download two open datasets.
- `.env` (not in git) sets `VERASONUS_DATA_ROOT=D:/verasonus-data`.
- FakeMusicCaps (Zenodo record 15063698) and FMA small were downloaded into `raw/fakemusiccaps/`
  and `raw/fma/`. Both matched their published checksums and were unzipped. The macOS metadata
  folder (`__MACOSX`) from the FakeMusicCaps zip was moved to the Recycle Bin.
- Claude decoded every clip with `load_audio`:
  - FakeMusicCaps: all 27,605 clips load (5 generators × 5,521; 16 kHz mono WAV, 10.0–10.24 s).
  - FMA small: 7,994 of 8,000 load. Six known-broken files (098565, 098567, 098569, 099134,
    108925, 133297) must be left out.
- FMA's metadata (`fma_metadata.zip`) was downloaded the same way. Its `tracks.csv` lists all
  8,000 FMA small clips with genre (8 genres × 1,000 clips), artist and licence.
- No audio went into git. In the repository, only this log changed.

**Left for the team.**
- All model, feature, augmentation and evaluation logic.
- Re-check dataset licences before downloading.
- Request access to Melody or Machine.
