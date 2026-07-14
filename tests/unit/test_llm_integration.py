from llm_layer.client import generate_medical_report
from llm_layer.client import evaluate_report

def test_evaluate_report_logic():
    # Teste de um laudo incompleto (deve retornar inválido)
    laudo_ruim = "Isso é apenas um teste curto."
    resultado = evaluate_report(laudo_ruim)
    assert resultado["valid"] is False 

    # Teste de um laudo completo (deve retornar válido)
    laudo_bom = "A área da lesão é 120mm², o risco é baixo. A conduta sugerida é acompanhamento rotineiro."
    resultado = evaluate_report(laudo_bom)
    assert resultado["valid"] is True

def test_generate_medical_report_uses_mocked_gemini(mock_gemini_response):
    patient_data = {
        "area_mean": 1200,
        "perimeter_mean": 150,
        "circularity": 0.7,
        "solidity": 0.8,
        "risk_score": 80,
        "risk_label": "alto",
    }

    response = generate_medical_report(patient_data)

    assert isinstance(response, dict)
    assert "report" in response
    assert "evaluation" in response
    assert "Laudo médico simulado" in response["report"]

