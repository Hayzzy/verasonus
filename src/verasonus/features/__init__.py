"""Feature extraction [aspect 4: intelligent sensing].

spectrograms  mel and constant-Q spectrograms for the deep models
handcrafted   MFCCs, spectral statistics, high-band energy, decoder-artefact features
embeddings    pretrained MERT and CLAP embeddings for probes
separation    optional Demucs stem separation
cache         compute per-window features once and save them under data/processed
"""
