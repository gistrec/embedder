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
    # Векторы нормализованы: [2, 1, 0] / sqrt(5) и [4, 1, 0] / sqrt(17).
    assert body["data"][0] == pytest.approx([2 / 5**0.5, 1 / 5**0.5, 0.0])
    assert body["data"][1] == pytest.approx([4 / 17**0.5, 1 / 17**0.5, 0.0])
    for vec in body["data"]:
        assert sum(x * x for x in vec) == pytest.approx(1.0)


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
    # Не-ASCII в заголовке — тоже 401, а не 500.
    weird = {"Authorization": "Bearer é".encode("latin-1")}
    assert client.post("/v1/embed", json={"input": ["a"]}, headers=weird).status_code == 401
    ok = {"Authorization": "Bearer s3cret"}
    assert client.post("/v1/embed", json={"input": ["a"]}, headers=ok).status_code == 200
    # health — без токена: его опрашивает мониторинг.
    assert client.get("/health").status_code == 200


def test_e5_prefixes_applied_by_type(fake):
    e = Embedder(name="intfloat/multilingual-e5-large", dim=3, _model=fake)
    e.embed(["вопрос"], "query")
    e.embed(["документ"], "document")
    assert fake.calls == [["query: вопрос"], ["passage: документ"]]


def test_prefix_lookup_ignores_case(fake):
    Embedder(name="Intfloat/Multilingual-E5-Large", dim=3, _model=fake).embed(["вопрос"], "query")
    assert fake.calls == [["query: вопрос"]]


def test_missing_prefixes_flags_only_unknown_models_that_need_them():
    from embedder.model import missing_prefixes

    need = "Text embeddings. Prefixes for queries/documents: necessary, 2024 year."
    no_need = "Text embeddings. Prefixes for queries/documents: not necessary, 2019 year."
    assert missing_prefixes("snowflake/snowflake-arctic-embed-m", need)
    assert not missing_prefixes("intfloat/multilingual-e5-large", need)  # есть в PREFIXES
    assert not missing_prefixes("fake/model", no_need)
    assert not missing_prefixes("fake/model", "описание без сведений о префиксах")


def test_empty_token_is_logged_at_startup(fake, monkeypatch, caplog):
    loader = lambda: Embedder(name="fake/model", dim=3, _model=fake)  # noqa: E731
    monkeypatch.setattr(settings, "token", "")
    with caplog.at_level("WARNING", logger="embedder"), TestClient(app_mod.create_app(loader=loader)):
        pass
    assert "EMBEDDER_TOKEN пуст" in caplog.text
    caplog.clear()
    monkeypatch.setattr(settings, "token", "s3cret")
    with caplog.at_level("WARNING", logger="embedder"), TestClient(app_mod.create_app(loader=loader)):
        pass
    assert "EMBEDDER_TOKEN пуст" not in caplog.text


def test_symmetric_model_has_no_prefix(fake):
    Embedder(name="fake/model", dim=3, _model=fake).embed(["текст"], "query")
    assert fake.calls == [["текст"]]
