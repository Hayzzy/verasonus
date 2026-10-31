from fastapi.testclient import TestClient

from verasonus import __version__
from verasonus.api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_analyze_rejects_bad_input_cleanly():
    # 503 without model files (as in CI), otherwise 422 because this is not audio.
    response = client.post("/analyze", files={"file": ("clip.wav", b"not real audio", "audio/wav")})
    assert response.status_code in (422, 503)
    assert isinstance(response.json()["detail"], str)
