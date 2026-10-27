"""Sensing front end [aspect 4]: decode, down-mix, resample, standardise and window audio.

audio_io     load / resample / save audio (implemented)
windowing    cut tracks into overlapping windows (implemented)
standardise  bring every clip to one format so models cannot learn file format (Phase 1)
"""
