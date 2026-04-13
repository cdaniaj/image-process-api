import pandas as pd
import os

def fill_out_csv(patient_name, patient_id, img_path, area, circularity, perimetro, bright_level, solidity):
    # Exemplo de como salvar no mesmo arquivo sem sobrescrever
    novo_dado = pd.DataFrame({
        "patient_name": [patient_name],
        "patient_id": [patient_id],
        "filename": [img_path],
        "area": [round(area, 3)],
        "circularity": [round(circularity, 3)],
        "perimeter": [round(perimetro, 3)],
        "nivel_brilho": [bright_level],
        "solidity": [round(solidity, 3)]
    })

    # append=True no modo 'a' (archive) do CSV
    novo_dado.to_csv('dataset_global.csv', mode='a', header=not os.path.exists('dataset_global.csv'), index=False)