# data/ (not in git)

Everything in this folder except this README is ignored by git. Datasets are large, and
many are copyrighted or licensed for non-commercial use only, so they never go into the
repository. See [docs/DATA.md](../docs/DATA.md).

Layout created by the scripts:

    raw/<source>/       downloads, untouched
    interim/<source>/   standardised audio (mono, target sample rate, FLAC)
    processed/          features, embeddings and caches
    metadata.csv        one row per clip

On Colab, set `VERASONUS_DATA_ROOT` to a Google Drive folder instead of using this folder.
