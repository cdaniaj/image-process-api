from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dto import PatientDiagnosticModel

import matplotlib.pyplot as plt
import seaborn as sns

modelo_rf = joblib.load("trained_model.pkl")
scaler = joblib.load('scaler.pkl')
modelo_lr = joblib.load('logistic_model.pkl')

def handleLearning():
    
    #Leitura e tratamento dos dados
    df = pd.read_csv('data.csv')
    
    
    cols_validar = ['area_mean', 'compactness_mean', 'perimeter_mean', 'concavity_mean', 'radius_mean']
    for col in cols_validar:
        mediana = df[col].median()
        df.loc[df[col] <= 0, col] = mediana
    
    df.drop_duplicates(inplace=True)

    X_completo = df[cols_validar]
    y_completo = df['diagnosis'].map({'M': 1, 'B': 0})
    
    X_train, X_test, y_train, y_test = train_test_split(X_completo, y_completo, test_size=0.2, random_state=42)   
     
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) 
    X_test_scaled = scaler.transform(X_test) 
    
    print("Média das features (deve ser ~0):", X_train_scaled.mean(axis=0))
    
    # 4. TREINAMENTO
    #random forest
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    rfc.fit(X_train_scaled, y_train)
    
    # logistic regression
    lr_model = LogisticRegression()
    lr_model.fit(X_train_scaled, y_train)
    
    # 5. VALIDAÇÃO
    y_pred = rfc.predict(X_test_scaled)
    y_pred_lr = lr_model.predict(X_test_scaled)
    
    print("\n=== RELATÓRIO DE CLASSIFICAÇÃO RANDOM FOREST ===")
    print(classification_report(y_test, y_pred))

    print("=== RELATÓRIO REGRESSÃO LOGÍSTICA ===")
    print(classification_report(y_test, y_pred_lr))


    print("=== MATRIZ DE CONFUSÃO ===")
    print(confusion_matrix(y_test, y_pred))
    
    # Para o Random Forest
    plt.figure(figsize=(5,4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão - Random Forest')
    plt.savefig('matriz_rf.png')

    # Para a Regressão Logística
    plt.figure(figsize=(5,4))
    sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap='Greens')
    plt.title('Matriz de Confusão - Regressão Logística')
    plt.savefig('matriz_lr.png') 
    
    # 6. EXPORTAÇÃO
    joblib.dump(rfc, 'trained_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(lr_model, 'logistic_model.pkl')
    print("\nArquivos salvos com sucesso: 'trained_model.pkl' e 'scaler.pkl'")
    
    # Extraindo a importância das características
    importances = rfc.feature_importances_
    feature_names = ['area_mean', 'compactness_mean', 'perimeter_mean', 'concavity_mean', 'radius_mean']
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Gerando o gráfico
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
    plt.title('Explicabilidade: Importância das Características (Random Forest)')
    plt.savefig('feature_importance.png')
    
    
def handlePrediction(patient_data: PatientDiagnosticModel):
    features = ['area_mean', 'compactness_mean', 'perimeter_mean', 'concavity_mean', 'radius_mean']
    
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
    
    consensus = "Divergente"
    if pred_rf == 1 and pred_lr == 1: consensus = "Maligno"
    if pred_rf == 0 and pred_lr == 0: consensus = "Benigno"
    
    coeficientes = pd.DataFrame(modelo_lr.coef_, columns=['area_mean', 'compactness_mean', 'perimeter_mean', 'concavity_mean', 'radius_mean'])
    print("Pesos da Regressão Logística:\n", coeficientes)
        
    return {
        "random_forest": {"prediction": int(pred_rf), "risk": round(prob_rf, 2)},
        "logistic_regression": {"prediction": int(pred_lr), "risk": round(prob_lr, 2)},
        "final_consensus": consensus
    }