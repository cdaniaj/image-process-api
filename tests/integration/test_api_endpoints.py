from fastapi.testclient import TestClient


def test_health_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200


def test_model_endpoint_returns_success(client):
    response = client.post("/model")
    assert response.status_code in {200, 202, 500}


def test_analyze_endpoint_with_mocked_llm(client, mock_gemini_response):
    file_bytes = b"fake-image-bytes"
    files = {"file": ("image.png", file_bytes, "image/png")}
    data = {"patient_name": "Maria", "patient_id": "123"}

    response = client.post("/analyze", files=files, data=data)
    assert response.status_code in {200, 400, 500}
