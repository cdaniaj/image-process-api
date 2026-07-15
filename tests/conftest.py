import io
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("GEMINI_API_KEY", "dummy-key")

from main import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def mock_gemini_response():
    class FakeResponse:
        text = "Laudo médico simulado. Área, risco e conduta foram avaliados."

    class FakeModel:
        def generate_content(self, *args, **kwargs):
            return FakeResponse()

    with patch("llm_layer.client.genai.GenerativeModel", return_value=FakeModel()):
        yield


@pytest.fixture
def mock_patient_payload():
    return {
        "name": "Maria Silva",
        "id": "12345",
        "file_name": "imagem.jpg",
        "area_mean": 1200.0,
        "compactness_mean": 0.45,
        "perimeter_mean": 150.0,
        "concavity_mean": 0.2,
        "radius_mean": 20.0,
        "finalConsensus": 1,
        "risk_score": 0.8,
        "risk_label": "alto",
        "prediction": 1,
        "prediction_lr": 1,
        "risk_score_lr": 0.7,
        "llm_explanation": "",
        "llm_insights": "",
    }


@pytest.fixture
def mock_data_file():
    mock_path = Path(__file__).parent / "mocks" / "mock_data.json"
    mock_path.write_text(json.dumps({"sample": "ok"}), encoding="utf-8")
    yield mock_path
    if mock_path.exists():
        mock_path.unlink()
