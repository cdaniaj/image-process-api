from fastapi.testclient import TestClient


def test_assistant_chat_returns_structured_response(monkeypatch):
    import main

    class FakeResponse:
        response = "Resposta de apoio."
        sources = ["data.csv"]
        disclaimer = "Sugestão de IA para auxílio médico. Validação humana obrigatória."
        session_id = "sessao-1"

        def model_dump(self):
            return self.__dict__

    monkeypatch.setattr(main, "run_assistant", lambda patient_id, query, session_id: FakeResponse())
    client = TestClient(main.app)

    response = client.post(
        "/assistant/chat",
        json={"patient_id": "123", "query": "Resuma", "session_id": "sessao-1"},
    )

    assert response.status_code == 200
    assert response.json()["disclaimer"] == "Sugestão de IA para auxílio médico. Validação humana obrigatória."
    assert response.json()["sources"] == ["data.csv"]


def test_assistant_chat_rejects_missing_fields():
    import main

    response = TestClient(main.app).post("/assistant/chat", json={"query": "Resuma"})

    assert response.status_code == 422


def test_assistant_chat_includes_synthetic_context():
    import main

    response = TestClient(main.app).post(
        "/assistant/chat",
        json={
            "patient_id": "PATIENT_001",
            "query": "Resuma a evolução",
            "session_id": "synthetic-session",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["context_used"]["patient_history"] is True
    assert payload["context_used"]["synthetic_history"] is True
    assert any(detail["type"] == "synthetic_history" for detail in payload["source_details"])