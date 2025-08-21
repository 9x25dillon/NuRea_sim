from fastapi.testclient import TestClient
from app.main import app

def test_prime():
    c = TestClient(app)
    r = c.post("/v1/prime", json={"query": "hello", "k": 5})
    assert r.status_code == 200
    body = r.json()
    assert "system" in body
    assert "messages" in body
