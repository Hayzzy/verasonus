"""VeraSonus: estimate whether a music track is AI-generated, show where, and explain why.

Package map, with the course aspect each part demonstrates in brackets:

    preprocessing, features   sensing: decoding, resampling, windowing, spectral features [4]
    augmentation              robustness suite and hybrid human + AI tracks
    models                    classical ML, deep models and embedding probes [1, 2]
    models.unsupervised       generator clustering and "unknown" outlier detection [1]
    ensemble                  model fusion, window-to-track aggregation, calibration [3]
    explain                   saliency maps and feature importance (sense-making) [4]
    evaluation                metrics, protocols, report figures and tables
    data                      dataset registry, metadata table, splits
    api                       FastAPI inference service

Aspects: [1] supervised/unsupervised learning, [2] machine/deep learning,
[3] hybrid/ensemble approaches, [4] intelligent sensing/sense-making.
"""

__version__ = "0.1.0"
