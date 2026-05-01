import pandas as pd
import os

def fill_out_csv(patient_name, patient_id, img_path, area_mean, compactness_mean, perimeter_mean, concavity_mean, radius_mean):
    # Exemplo de como salvar no mesmo arquivo sem sobrescrever
    novo_dado = pd.DataFrame({
        "patient_name": [patient_name],
        "patient_id": [patient_id],
        "filename": [img_path],
        "area_mean": [round(area_mean, 3)],
        "compactness_mean": [round(compactness_mean, 3)],
        "perimeter_mean": [round(perimeter_mean, 3)],
        "concavity_mean": [round(concavity_mean, 3)],
        "radius_mean": [round(radius_mean, 3)]
    })

    # append=True no modo 'a' (archive) do CSV
    novo_dado.to_csv('dataset_global1.csv', mode='a', header=not os.path.exists('dataset_global1.csv'), index=False)