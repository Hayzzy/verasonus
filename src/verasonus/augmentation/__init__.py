"""Robustness suite (configs/augment.yaml) and hybrid human + AI tracks.

codecs      MP3 / AAC / Opus re-encoding at several bitrates (ffmpeg)
transforms  pitch shift, time stretch, added noise, loudness normalisation
playback    simulated phone-speaker playback and re-recording
hybrid      chosen stems passed through a neural codec to build partly-AI tracks
"""
