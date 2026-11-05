import pytest
from fastapi.testclient import TestClient

from embedder import app as app_mod
from embedder.config import settings
from embedder.model import Embedder


class FakeModel:
    def __init__(self):
        self.calls = []

    def embed(self, inputs, batch_size=32):
        import numpy as np

        self.calls.append(list(inputs))
        return [np.array([float(len(t)), 1.0, 0.0]) for t in inputs]


@pytest.fixture
def fake():
    return FakeModel()


@pytest.fixture
def client(fake, monkeypatch):
    monkeypatch.setattr(settings, "token", "")
    app = app_mod.create_app(loader=lambda: Embedder(name="fake/model", dim=3, _model=fake))
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model": "fake/model", "dim": 3}


def test_embed_returns_vector_per_input(client):
    r = client.post("/v1/embed", json={"input": ["ab", "abcd"], "type": "query"})
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "fake/model" and body["dim"] == 3
    assert body["data"] == [[2.0, 1.0, 0.0], [4.0, 1.0, 0.0]]


def test_limits(client, monkeypatch):
    monkeypatch.setattr(settings, "max_inputs", 2)
    monkeypatch.setattr(settings, "max_chars", 5)
    assert client.post("/v1/embed", json={"input": ["a", "b", "c"]}).status_code == 413
    assert client.post("/v1/embed", json={"input": ["123456"]}).status_code == 413
    assert client.post("/v1/embed", json={"input": ["  "]}).status_code == 422
    assert client.post("/v1/embed", json={"input": []}).status_code == 422
    assert client.post("/v1/embed", json={"input": ["a"], "type": "other"}).status_code == 422


def test_token_required_when_set(client, monkeypatch):
    monkeypatch.setattr(settings, "token", "s3cret")
    assert client.post("/v1/embed", json={"input": ["a"]}).status_code == 401
    bad = {"Authorization": "Bearer nope"}
    assert client.post("/v1/embed", json={"input": ["a"]}, headers=bad).status_code == 401
    ok = {"Authorization": "Bearer s3cret"}
    assert client.post("/v1/embed", json={"input": ["a"]}, headers=ok).status_code == 200
    # health — без токена: его опрашивает мониторинг.
    assert client.get("/health").status_code == 200


def test_e5_prefixes_applied_by_type(fake):
    e = Embedder(name="intfloat/multilingual-e5-large", dim=3, _model=fake)
    e.embed(["вопрос"], "query")
    e.embed(["документ"], "document")
    assert fake.calls == [["query: вопрос"], ["passage: документ"]]


def test_symmetric_model_has_no_prefix(fake):
    Embedder(name="fake/model", dim=3, _model=fake).embed(["текст"], "query")
    assert fake.calls == [["текст"]]
