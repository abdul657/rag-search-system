from fastapi.testclient import TestClient

from main import app


class FakePipeline:
    def index_documents(self, documents):
        return {"status": "success", "indexed": len(documents)}

    def search(self, query):
        return {
            "query": query.text,
            "results": [
                {
                    "id": "doc-1",
                    "text": "FastAPI is a Python web framework.",
                    "score": 0.91,
                    "metadata": {"source": "test"},
                }
            ],
        }


def client_with_fake_pipeline(monkeypatch):
    monkeypatch.setattr(app.state, "pipeline", FakePipeline(), raising=False)
    return TestClient(app)


def test_health_does_not_require_external_services():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_documents(monkeypatch):
    client = client_with_fake_pipeline(monkeypatch)
    response = client.post(
        "/index",
        json=[{"id": "doc-1", "text": "FastAPI is a Python web framework."}],
    )
    assert response.status_code == 201
    assert response.json() == {"status": "success", "indexed": 1}


def test_index_rejects_empty_list(monkeypatch):
    client = client_with_fake_pipeline(monkeypatch)
    response = client.post("/index", json=[])
    assert response.status_code == 400


def test_search_documents(monkeypatch):
    client = client_with_fake_pipeline(monkeypatch)
    response = client.post("/search", json={"text": "web framework", "top_k": 3})
    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "web framework"
    assert payload["results"][0]["id"] == "doc-1"
    assert payload["results"][0]["score"] == 0.91


def test_search_validates_top_k(monkeypatch):
    client = client_with_fake_pipeline(monkeypatch)
    response = client.post("/search", json={"text": "query", "top_k": 0})
    assert response.status_code == 422
