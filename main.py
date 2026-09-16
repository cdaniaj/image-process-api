from pathlib import Path
from dotenv import load_dotenv


env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Form
from pydantic import BaseModel
from fastapi.responses import JSONResponse


import threading
from pathlib import Path
from prometheus_fastapi_instrumentator import Instrumentator

from contours import get_contours, get_contours_data
from normalization import normalize_image
from morphology import apply_morphology

from data_extraction.extraction import fill_out_csv
from outputfile.outputfile import get_file

from ai_model.mamography_rf.model import handleLearning, handlePrediction
from dto import convert_to_dto, get_risk_label, PatientDiagnosticModel
from reports.diagram import getReports

from llm_layer.client import generate_medical_report
from langchain_agent.agent import AssistantResponse, run_assistant
from observability.logger import logger

import cv2 as cv
import numpy as np

app = FastAPI(title="Image process API")
training_lock = threading.Lock()
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024
ASSISTANT_DISCLAIMER = "Sugestão de IA para auxílio médico. Validação humana obrigatória."


class AssistantChatRequest(BaseModel):
    patient_id: str
    query: str
    session_id: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:4200"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health_check():
    return {"status": "ok"}

@app.post("/reports")
async def get_reports():
    getReports()
    return {"status": "success"}


@app.post("/assistant/chat", response_model=AssistantResponse)
async def assistant_chat(payload: AssistantChatRequest):
    """Responde perguntas contextualizadas com histórico e protocolos locais."""
    try:
        return run_assistant(payload.patient_id, payload.query, payload.session_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro no assistente: {str(exc)}") from exc

@app.post("/confirm", summary="Confirma os dados extraídos de uma amostra e os salva em um arquivo CSV para treinamento futuro.")
async def postAISample(patient_data: PatientDiagnosticModel):
    logger.info("Amostra confirmada para patient_id=%s", patient_data.id)
    try:
        fill_out_csv(
            patient_data.name, 
            patient_data.id, 
            patient_data.file_name, 
            patient_data.area_mean,
            patient_data.compactness_mean, 
            patient_data.perimeter_mean, 
            patient_data.concavity_mean, 
            patient_data.radius_mean,
            patient_data.finalConsensus,
            patient_data.risk_score,
            patient_data.risk_label,
            patient_data.prediction,
            patient_data.prediction_lr,
            patient_data.risk_score_lr,
        )
        return {
            "status": "success"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro: {str(e)}")
        
        
@app.post("/model", summary="Treina modelos de machine learning com os dados disponíveis.")
async def postModel(background_tasks: BackgroundTasks):
    """
        Este endpoint é responsável por treinar os modelos de machine learning utilizando os dados disponíveis. 
        Ele pode ser acionado após a coleta de um número suficiente de amostras confirmadas para atualizar os modelos com 
        as informações mais recentes. 
        
        O processo inclui:
        
        1. Leitura dos dados de treinamento a partir do arquivo CSV.
        
        2. Pré-processamento dos dados para garantir que estejam no formato adequado para o treinamento.
        
        3. Treinamento dos modelos de machine learning, como Random Forest e Logistic Regression.
        
        4. Validação dos modelos utilizando um conjunto de teste para avaliar seu desempenho.
        
        5. Exportação dos modelos treinados para arquivos que podem ser utilizados posteriormente para fazer predições em novas amostras.
    """
    try:
        if not training_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="Já existe um treinamento em execução.")

        def run_training():
            try:
                handleLearning()
            finally:
                training_lock.release()

        try:
            background_tasks.add_task(run_training)
        except Exception:
            training_lock.release()
            raise
        return JSONResponse(
            status_code=202,
            content={
                "status": "training_started",
                "detail": "Otimização via AG iniciada em background. Monitore os logs do contêiner para acompanhar as gerações."
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro: {str(e)}")

@app.post("/analyze", summary="Analisa uma amostra de célula e retorna os dados extraídos e a previsão de risco.")
async def analyze_cell(
    file: UploadFile = File(..., description="Imagem da amostra de célula a ser analisada. Formato suportado: JPG, PNG."), 
    patient_name: str = Form(..., description="Nome do paciente associado à amostra."),
    patient_id: str = Form(..., description="ID do paciente associado à amostra.")):
    """
     Este endpoint recebe uma imagem de célula, processa-a para extrair características relevantes, e então utiliza um modelo de machine learning para prever o risco associado à amostra. O processo inclui:
     
     1. Leitura e decodificação da imagem.
     
     2. Normalização da imagem para melhorar a qualidade dos dados.
     
     3. Aplicação de filtros para reduzir ruídos.
     
     4. Binarização para segmentar a lesão.
     
     5. Limpeza morfológica para refinar a segmentação. 
     
     6. Extração de contornos e cálculo de características como área, perímetro, circularidade e solidez.
     
     7. Utilização de um modelo de machine learning para prever o risco com base nas características extraídas.
     
        em seguida, retorna os dados extraídos, a previsão de risco e outras informações relevantes para o cliente. 
    """
    try:
        if file.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=415,
                detail="Tipo de imagem não suportado. Use image/jpeg ou image/png.",
            )

        contents = await file.read()
        if len(contents) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="A imagem excede o limite de 10 MB.")

        nparr = np.frombuffer(contents, dtype=np.uint8)
        img = cv.imdecode(nparr, cv.IMREAD_GRAYSCALE)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Imagem inválida ou não decodificável.")

        # 1. RECORTE (ROI)
        amostra = img

        # 2. NORMALIZAÇÃO
        amostra = normalize_image(amostra, 85, 100)

        # 3. PRÉ-PROCESSAMENTO
        amostra = cv.medianBlur(amostra, 5)
        
        # CLAHE
        clahe = cv.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        amostra = clahe.apply(amostra)
        
        # 4. BINARIZAÇÃO
        ret_calculado, thresh = cv.threshold(amostra, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU) 

        # 5. LIMPEZA MORFOLÓGICA
        thresh = apply_morphology(thresh)

        # 6. EXTRAÇÃO DO CONTORNO
        contours = get_contours(thresh)

        if not contours:
            raise HTTPException(status_code=422, detail="Nenhum contorno foi encontrado na imagem.")

        contours_data = get_contours_data(contours, amostra.shape)
        if contours_data["train_data"].get("max_contour") is None:
            raise HTTPException(status_code=422, detail="Nenhum contorno válido foi encontrado na imagem.")

        safe_filename = Path(file.filename or "amostra.png").name
        patient_data = convert_to_dto(
            patient_name,
            patient_id,
            safe_filename,
            contours_data["train_data"]["area_mean"],
            contours_data["train_data"]["compactness_mean"],
            contours_data["train_data"]["perimeter_mean"],
            contours_data["train_data"]["concavity_mean"],
            contours_data["train_data"]["radius_mean"],
            0,
        )
        predictionData = handlePrediction(patient_data)
        patient_data.risk_score = predictionData["random_forest"]["risk"]
        patient_data.prediction = predictionData["random_forest"]["prediction"]
        patient_data.prediction_lr = predictionData["logistic_regression"]["prediction"]
        patient_data.risk_score_lr = predictionData["logistic_regression"]["risk"]
        patient_data.risk_label = get_risk_label(patient_data.risk_score)
        patient_data.finalConsensus = predictionData["final_consensus"]

        llm_result = generate_medical_report(patient_data.model_dump())
        patient_data.llm_explanation = llm_result["report"]
        logger.info("Qualidade do laudo: %s", llm_result["evaluation"])

        get_file(
            patient_data.file_name,
            amostra,
            contours_data["train_data"]["max_contour"],
        )
  
        return {
            "name": patient_data.name,
            "id": patient_data.id,
            "area": round(contours_data["interface_data"]["area"], 2),
            "perimeter": round(contours_data["interface_data"]["perimeter"], 2),
            "circularity": round(contours_data["interface_data"]["circularity"], 2),
            "solidity": round(contours_data["interface_data"]["solidity"], 2),
            "risk_score": patient_data.risk_score,
            "risk_label": patient_data.risk_label,
            "patient_data": patient_data
        }
        

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro: {str(e)}")
    finally:
        await file.close()
        
Instrumentator().instrument(app).expose(app)
