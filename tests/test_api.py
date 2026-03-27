# We use FastAPI's built-in test client — no need to run the server
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_rejects_non_pdf():
    # Our /upload should only accept PDFs
    # We send a .txt file and expect a 400 error back
    response = client.post(
        "/upload",
        files={"file": ("notes.txt", b"some text content", "text/plain")}
    )
    assert response.status_code == 400


def test_ask_rejects_empty_question():
    # Empty questions should be rejected before hitting the LLM
    # This saves unnecessary API calls to Groq
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 400


def test_ask_rejects_whitespace_question():
    # "   " is technically not empty but also not a real question
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 400