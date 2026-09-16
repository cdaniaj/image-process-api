import json

from langchain_agent import tools


def test_search_patient_history_reads_confirmed_dataset(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset_extraction.csv"
    dataset.write_text(
        "patient_id,area_mean,risk_score,diagnosis\n123,10,80,1\n999,20,5,0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(tools, "CONFIRMED_DATASET", dataset)
    monkeypatch.setattr(tools, "LEGACY_DATASET", tmp_path / "missing.csv")

    result = tools.search_patient_history.invoke("123")
    payload = json.loads(result)

    assert len(payload["records"]) == 1
    assert payload["records"][0]["risk_score"] == 80
    assert payload["sources"]


def test_search_medical_protocols_warns_when_directory_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "PROTOCOLS_DIR", tmp_path / "empty")

    payload = json.loads(tools.search_medical_protocols.invoke("conduta"))

    assert payload["results"] == []
    assert "Nenhum protocolo" in payload["warning"]


def test_search_patient_history_reads_synthetic_history(monkeypatch):
    monkeypatch.setattr(tools, "LEGACY_DATASET", tools.BASE_DIR / "missing.csv")
    monkeypatch.setattr(tools, "CONFIRMED_DATASET", tools.BASE_DIR / "missing-confirmed.csv")

    payload = json.loads(tools.search_patient_history.invoke("PATIENT_001"))

    assert payload["record_count"] == 3
    assert payload["records"][0]["exam_date"] == "2025-01-15"
    assert any(source["type"] == "synthetic_history" for source in payload["source_details"])


def test_run_assistant_always_returns_disclaimer_and_sources(monkeypatch):
    monkeypatch.setattr(tools, "LEGACY_DATASET", tools.BASE_DIR / "missing.csv")
    monkeypatch.setattr(tools, "CONFIRMED_DATASET", tools.BASE_DIR / "missing-confirmed.csv")
    monkeypatch.setattr(tools, "PROTOCOLS_DIR", tools.BASE_DIR / "missing-protocols")

    from langchain_agent.agent import DISCLAIMER, run_assistant

    result = run_assistant("123", "Qual o histórico?", "test-session")

    assert result.disclaimer == DISCLAIMER
    assert result.session_id == "test-session"
    assert result.response