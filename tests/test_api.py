from fastapi.testclient import TestClient

from verasonus import __version__
from verasonus.api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_analyze_is_not_available_yet():
    response = client.post("/analyze", files={"file": ("clip.wav", b"not real audio", "audio/wav")})
    assert response.status_code == 501
