from pydantic import BaseModel

class PatientDiagnosticModel(BaseModel):
    name: str
    file_name: str
    id: str
    area_mean: float
    perimeter_mean: float
    compactness_mean: float
    concavity_mean: float
    radius_mean: float
    risk_score: float
    risk_label: str
    prediction: int
    prediction_lr: int
    risk_score_lr: float
    
def convert_to_dto(
    patient_name, 
    patient_id, 
    img_path, 
    area_mean, 
    compactness_mean, 
    perimeter_mean, 
    concavity_mean, 
    radius_mean
):
    patient_data = PatientDiagnosticModel(
        id=patient_id,
        name=patient_name,
        file_name=img_path,
        area_mean=area_mean,
        compactness_mean=compactness_mean,
        perimeter_mean=perimeter_mean,
        concavity_mean=concavity_mean,
        radius_mean=radius_mean,
        risk_label='',
        risk_score=0,
        prediction=0,
        prediction_lr=0,
        risk_score_lr=0
    )

    return patient_data


def get_risk_label(risk_score: float):
    if risk_score <= 30:
        return "no_risk"
    
    if risk_score >= 31 and risk_score <= 60:
        return "moderate_risk"
    
    if risk_score >= 61:
        return "high_risk"