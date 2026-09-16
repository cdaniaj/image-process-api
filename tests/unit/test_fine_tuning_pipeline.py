import json
from pathlib import Path

from fine_tunning.generate_dataset import build_risk_response
from fine_tunning.prepare_dataset import anonymize_text, build_dataset, sanitize_record


def test_build_risk_response_high_risk_mentions_protocol_steps():
    response = build_risk_response("risco elevado", 85.6, 0.77, 1774.1)

    assert "avaliação clínica completa" in response.lower()
    assert "ultrassonografia complementar" in response.lower()
    assert "biópsia" in response.lower()
    assert "validação médica" in response.lower()
    assert "neoadjuvância" not in response.lower()


def test_anonymize_text_replaces_pii_and_preserves_medical_context():
    text = "Paciente Maria da Silva, CPF 123.456.789-09, tel 11999999999. Diagnóstico: risco moderado."

    result = anonymize_text(text)

    assert "Maria" not in result
    assert "Silva" not in result
    assert "123.456.789-09" not in result
    assert "11999999999" not in result
    assert "PACIENTE_HASH_" in result
    assert "risco moderado" in result.lower()


def test_build_dataset_creates_jsonl_with_messages(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "name,id,area_mean,perimeter_mean,compactness_mean,concavity_mean,radius_mean,diagnosis\n"
        "Maria Silva,12345,1200,150,0.45,0.2,20,1\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "dataset_anonimizado.jsonl"
    build_dataset(csv_path=csv_path, output_path=output_path, limit=1)

    assert output_path.exists()
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) >= 1
    assert "messages" in rows[0]
    assert rows[0]["messages"][0]["role"] == "system"
    assert rows[0]["messages"][1]["role"] == "user"


def test_sanitize_record_replaces_patient_fields():
    record = {
        "name": "Maria Silva",
        "cpf": "123.456.789-09",
        "diagnosis": "risco moderado",
    }

    sanitized = sanitize_record(record)

    assert sanitized["name"].startswith("PACIENTE_HASH_")
    assert sanitized["cpf"] == "CPF_REMOVIDO"
    assert sanitized["diagnosis"] == "risco moderado"
