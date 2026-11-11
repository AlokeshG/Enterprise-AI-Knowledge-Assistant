"""
Basic smoke tests.

Note: full integration tests that hit OpenAI are intentionally excluded
from CI by default since they require a paid API key. These tests cover
what can be verified without live API calls.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_list_documents_empty_or_valid_shape():
    response = client.get("/api/documents")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_upload_rejects_unsupported_file_type():
    files = [("files", ("malware.exe", b"binary-content", "application/octet-stream"))]
    response = client.post("/api/upload", files=files)
    # Upload endpoint processes per-file; unsupported type surfaces as a 400
    # from the exception handler once save_and_index raises.
    assert response.status_code in (400, 500)


def test_query_validation_rejects_empty_question():
    response = client.post("/api/query", json={"question": ""})
    assert response.status_code == 422
