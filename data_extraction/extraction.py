import pandas as pd
import os
import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
CSV_FOLDER = Path(os.getenv("IMAGE_PROCESS_DATA_DIR", BASE_DIR / "data_extraction" / "gerados"))
CONFIRMED_DATASET_PATH = CSV_FOLDER / "dataset_extraction.csv"
CONFIRMED_COLUMNS = [
    "patient_name",
    "patient_id",
    "filename",
    "area_mean",
    "compactness_mean",
    "perimeter_mean",
    "concavity_mean",
    "radius_mean",
    "diagnosis",
    "risk_score",
    "risk_label",
    "prediction",
    "prediction_lr",
    "risk_score_lr",
]


def _read_csv_compat(path: Path) -> pd.DataFrame:
    """Lê o CSV confirmado mesmo quando contém linhas do formato legado."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        return pd.DataFrame(columns=CONFIRMED_COLUMNS)

    header = [column.strip() for column in rows[0]]
    records = []
    for row in rows[1:]:
        if not row:
            continue
        record = {
            column: value
            for column, value in zip(header, row)
            if column in CONFIRMED_COLUMNS
        }
        if header[:9] == CONFIRMED_COLUMNS[:9] and len(row) > 9:
            record.update(dict(zip(CONFIRMED_COLUMNS[9:], row[9:])))
        records.append({column: record.get(column) for column in CONFIRMED_COLUMNS})

    data = pd.DataFrame(records, columns=CONFIRMED_COLUMNS)
    numeric_columns = [
        "area_mean",
        "compactness_mean",
        "perimeter_mean",
        "concavity_mean",
        "radius_mean",
        "diagnosis",
        "risk_score",
        "prediction",
        "prediction_lr",
        "risk_score_lr",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def _migrate_confirmed_dataset() -> None:
    if not CONFIRMED_DATASET_PATH.exists():
        return
    data = _read_csv_compat(CONFIRMED_DATASET_PATH)
    data.to_csv(CONFIRMED_DATASET_PATH, index=False)

def fill_out_csv(
    patient_name,
    patient_id,
    img_path,
    area_mean,
    compactness_mean,
    perimeter_mean,
    concavity_mean,
    radius_mean,
    finalConsensus,
    risk_score=None,
    risk_label=None,
    prediction=None,
    prediction_lr=None,
    risk_score_lr=None,
):
    CSV_FOLDER.mkdir(parents=True, exist_ok=True)
    _migrate_confirmed_dataset()
    
    novo_dado = pd.DataFrame({
        "patient_name": [patient_name],
        "patient_id": [patient_id],
        "filename": [img_path],
        "area_mean": [round(area_mean, 3)],
        "compactness_mean": [round(compactness_mean, 3)],
        "perimeter_mean": [round(perimeter_mean, 3)],
        "concavity_mean": [round(concavity_mean, 3)],
        "radius_mean": [round(radius_mean, 3)],
        "diagnosis": [finalConsensus],
        "risk_score": [risk_score],
        "risk_label": [risk_label],
        "prediction": [prediction],
        "prediction_lr": [prediction_lr],
        "risk_score_lr": [risk_score_lr],
    })

    novo_dado.to_csv(
        CONFIRMED_DATASET_PATH,
        mode='a',
        header=not CONFIRMED_DATASET_PATH.exists(),
        index=False,
    )


def load_training_data() -> pd.DataFrame:
    """Combina o dataset legado com as amostras confirmadas pela API."""
    legacy_path = Path(os.getenv("IMAGE_PROCESS_LEGACY_DATASET", BASE_DIR / "data.csv"))
    frames = []

    if legacy_path.exists():
        frames.append(pd.read_csv(legacy_path))
    if CONFIRMED_DATASET_PATH.exists():
        frames.append(_read_csv_compat(CONFIRMED_DATASET_PATH))

    if not frames:
        raise FileNotFoundError("Nenhum dataset de treinamento foi encontrado.")

    data = pd.concat(frames, ignore_index=True, sort=False)
    required = [
        "area_mean",
        "compactness_mean",
        "perimeter_mean",
        "concavity_mean",
        "radius_mean",
        "diagnosis",
    ]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes no dataset: {', '.join(missing)}")

    return data[required + [column for column in ("patient_id", "patient_name", "filename") if column in data.columns]]