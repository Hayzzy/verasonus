"""Data layer: dataset registry, per-clip metadata table and train/val/test splits.

sources   registry of the data sources in configs/datasets.yaml; download/scan stubs
metadata  the per-clip metadata table schema
splits    track-grouped and generator-held-out splits (implemented)
"""
