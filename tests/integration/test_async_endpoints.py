from fastapi.testclient import TestClient
from main import app
from unittest.mock import patch

def test_model_endpoint_is_async():
    """
    Valida que o endpoint /model retorna 202 Accepted (assíncrono) 
    e não bloqueia a execução.
    """
    client = TestClient(app)
    
    # Mockando a tarefa de background para não treinar de verdade durante o teste
    with patch("fastapi.background.BackgroundTasks.add_task") as mock_task:
        response = client.post("/model")
        
        # O status 202 indica que a tarefa foi aceita e disparada em background
        assert response.status_code == 202
        assert response.json()["status"] == "training_started"
        
        # Verifica se o FastAPI realmente chamou o add_task
        mock_task.assert_called_once()