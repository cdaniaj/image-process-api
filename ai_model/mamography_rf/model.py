from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dto import PatientDiagnosticModel

modelo_rf = joblib.load("trained_model.pkl")
scaler = joblib.load('scaler.pkl')
modelo_lr = joblib.load('logistic_model.pkl')

def handleLearning():
    # 1. PREPARAÇÃO
    df = pd.read_csv('data.csv')
    
    # DICA: Removi o nome x_train aqui para não confundir com a saída do train_test_split
    X_completo = df[['area_mean', 'compactness_mean', 'perimeter_mean', 'concavity_mean', 'radius_mean']]
    y_completo = df['diagnosis'].map({'M': 1, 'B': 0})
    
    # 2. DIVISÃO
    # O test_size=0.2 garante os 20% para validação exigidos no PDF
    X_train, X_test, y_train, y_test = train_test_split(X_completo, y_completo, test_size=0.2, random_state=42)   
     
    # 3. ESCALONAMENTO
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) 
    X_test_scaled = scaler.transform(X_test) 
    
    # Isso deve imprimir valores muito próximos de zero (ex: 1.2e-16)
    print("Média das features (deve ser ~0):", X_train_scaled.mean(axis=0))
    
    # 4. TREINAMENTO
    #random forest
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    rfc.fit(X_train_scaled, y_train)
    
    # logistic regression
    lr_model = LogisticRegression()
    lr_model.fit(X_train_scaled, y_train)
    
    # 5. VALIDAÇÃO
    # CORREÇÃO AQUI: Você chamou de 'rfc', então deve usar 'rfc.predict'
    y_pred = rfc.predict(X_test_scaled)
    y_pred_lr = lr_model.predict(X_test_scaled)
    
    print("\n=== RELATÓRIO DE CLASSIFICAÇÃO RANDOM FOREST ===")
    print(classification_report(y_test, y_pred))

    print("=== RELATÓRIO REGRESSÃO LOGÍSTICA ===")
    print(classification_report(y_test, y_pred_lr))


    print("=== MATRIZ DE CONFUSÃO ===")
    print(confusion_matrix(y_test, y_pred))
    
    # 6. EXPORTAÇÃO
    joblib.dump(rfc, 'trained_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(lr_model, 'logistic_model.pkl')
    print("\nArquivos salvos com sucesso: 'trained_model.pkl' e 'scaler.pkl'")
    
def handlePrediction(patient_data: PatientDiagnosticModel):
    # 1. Definimos os nomes das colunas para evitar o UserWarning
    features = ['area_mean', 'compactness_mean', 'perimeter_mean', 'concavity_mean', 'radius_mean']
    
    # 2. Criamos um DataFrame (isso garante a ordem correta das colunas)
    data_df = pd.DataFrame([[
        patient_data.area_mean,
        patient_data.compactness_mean, 
        patient_data.perimeter_mean,
        patient_data.concavity_mean,
        patient_data.radius_mean
    ]], columns=features)
    
    # 3. Escalonamento usando o DataFrame
    scaledData = scaler.transform(data_df)
        
    # 4. Predições
    prob_rf = modelo_rf.predict_proba(scaledData)[0][1] * 100
    pred_rf = modelo_rf.predict(scaledData)[0]
    
    prob_lr = modelo_lr.predict_proba(scaledData)[0][1] * 100
    pred_lr = modelo_lr.predict(scaledData)[0]
    
    # 5. Lógica de Consenso (Opcional, mas excelente para o PDF)
    # Se um diz 1 e outro diz 0, o risco é inconclusivo
    consensus = "Divergente"
    if pred_rf == 1 and pred_lr == 1: consensus = "Maligno"
    if pred_rf == 0 and pred_lr == 0: consensus = "Benigno"
    
    # Veja quais características a LR mais valoriza
    coeficientes = pd.DataFrame(modelo_lr.coef_, columns=['area_mean', 'compactness_mean', 'perimeter_mean', 'concavity_mean', 'radius_mean'])
    print("Pesos da Regressão Logística:\n", coeficientes)
        
    return {
        "random_forest": {"prediction": int(pred_rf), "risk": round(prob_rf, 2)},
        "logistic_regression": {"prediction": int(pred_lr), "risk": round(prob_lr, 2)},
        "final_consensus": consensus
    }