from unittest.mock import patch

import numpy as np
import pandas as pd

from ai_model.mamography_rf import model as model_module
from dto import PatientDiagnosticModel


def test_models_exist_returns_false_when_files_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert model_module.models_exist() is False


def test_handle_prediction_returns_expected_shape():
    patient = PatientDiagnosticModel(
        name="Maria",
        id="123",
        file_name="img.png",
        area_mean=100.0,
        compactness_mean=0.4,
        perimeter_mean=50.0,
        concavity_mean=0.2,
        radius_mean=10.0,
        risk_score=0.0,
        risk_label="",
        prediction=0,
        prediction_lr=0,
        risk_score_lr=0.0,
        finalConsensus=1,
        llm_explanation="",
        llm_insights="",
    )

    fake_model = type("FakeModel", (), {})
    fake_model.predict_proba = lambda self, X: np.array([[0.2, 0.8]])
    fake_model.predict = lambda self, X: np.array([1])
    fake_model.coef_ = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])

    model_module.modelo_rf = fake_model()
    model_module.modelo_lr = fake_model()
    model_module.scaler = type("Scaler", (), {"transform": lambda self, X: X})()

    result = model_module.handlePrediction(patient)

    assert result["random_forest"]["prediction"] == 1
    assert result["random_forest"]["risk"] == 80.0
    assert result["logistic_regression"]["prediction"] == 1
    assert result["logistic_regression"]["risk"] == 80.0
    assert result["final_consensus"] in {0, 1, 2}
