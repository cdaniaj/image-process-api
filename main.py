from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi import Form

from contours import get_contours, get_contours_data
from outputfile import get_file
from normalization import normalize_image
from morphology import apply_morphology
from data_extraction.extraction import fill_out_csv

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_cell(
    file: UploadFile = File(...), 
    patient_name: str = Form(...),
    patient_id: str = Form(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, dtype=np.uint8)
        img = cv.imdecode(nparr, cv.IMREAD_GRAYSCALE)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Imagem inválida.")

        # 1. RECORTE (ROI)
        amostra = img

        # 2. NORMALIZAÇÃO MAIS SEGURA
        # podemos ser mais permissivos aqui para não deletar a borda da lesão.
        amostra = normalize_image(amostra, 85, 100)

        # 3. PRÉ-PROCESSAMENTO
        amostra = cv.medianBlur(amostra, 5)
        
        # CLAHE levemente ajustado
        clahe = cv.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        amostra = clahe.apply(amostra)
        
        # 4. BINARIZAÇÃO
        ret_calculado, thresh = cv.threshold(amostra, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU) 

        # 5. LIMPEZA MORFOLÓGICA
        thresh = apply_morphology(thresh)

        # 6. EXTRAÇÃO DO CONTORNO
        contours = get_contours(thresh)

        if contours:
            contours_data = get_contours_data(contours, amostra.shape)

            # Visualização de Debug
            get_file(
                file.filename, 
                amostra, 
                contours_data["max_contour"]
            )

        fill_out_csv(
            patient_name, 
            patient_id,
            file.filename, 
            contours_data["area"], 
            contours_data["circularity"], 
            contours_data["perimetro"],
            "N/A", 
            contours_data["solidity"]
        )
        # ... resto do return e except continuam iguais ...
        return {
            "name": patient_name,
            "id": patient_id,
            "area": round(contours_data["area"], 2),
            "perimeter": round(contours_data["perimetro"], 2),
            "circularity": round(contours_data["circularity"], 2),
            "solidity": round(contours_data["solidity"], 2)
        }
        

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro: {str(e)}")
    finally:
        await file.close()