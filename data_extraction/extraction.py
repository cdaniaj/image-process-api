import pandas as pd
import os

# Caminho relativo que funciona localmente e em Docker
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_FOLDER = os.path.join(BASE_DIR, "data_extraction", "gerados")

def fill_out_csv(patient_name, patient_id, img_path, area_mean, compactness_mean, perimeter_mean, concavity_mean, radius_mean, finalConsensus):
    os.makedirs(CSV_FOLDER, exist_ok=True)
    
    novo_dado = pd.DataFrame({
        "patient_name": [patient_name],
        "patient_id": [patient_id],
        "filename": [img_path],
        "area_mean": [round(area_mean, 3)],
        "compactness_mean": [round(compactness_mean, 3)],
        "perimeter_mean": [round(perimeter_mean, 3)],
        "concavity_mean": [round(concavity_mean, 3)],
        "radius_mean": [round(radius_mean, 3)],
        "diagnosis": [finalConsensus]
    })

    csv_path = os.path.join(CSV_FOLDER, 'dataset_extraction.csv')
    novo_dado.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)