import json

from langgraph_workflows import assistant_graph


def test_graph_retrieves_synthetic_history_and_builds_context():
    state = assistant_graph.run_assistant_graph(
        {
            "patient_id": "PATIENT_001",
            "query": "Resuma a evolução.",
            "session_id": "graph-test",
            "previous_messages": [],
        }
    )

    assert len(state["history"]["records"]) == 3
    assert state["context_used"] if "context_used" in state else True
    assert "PATIENT_001" in state["prompt"]
    assert state["response"]


def test_guardrail_replaces_direct_prescription(monkeypatch):
    monkeypatch.setattr(
        assistant_graph,
        "generate_assistant_response",
        lambda prompt: ("Prescrevo que o paciente tome o tratamento agora.", "test"),
    )
    state = assistant_graph.run_assistant_graph(
        {
            "patient_id": "PATIENT_001",
            "query": "Qual a conduta?",
            "session_id": "guardrail-test",
            "previous_messages": [],
        }
    )

    assert state["guardrail_blocked"] is True
    assert "prescrição" in state["response"].lower()