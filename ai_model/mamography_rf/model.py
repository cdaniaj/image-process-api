from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dto import PatientDiagnosticModel

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import time
from observability.logger import logger
from ai_model.mamography_rf.genetic_optimizer import GeneticOptimizer


MODEL_PATHS = {
    "rf": "trained_model.pkl",
    "scaler": "scaler.pkl",
    "lr": "logistic_model.pkl",
}


def _resolve_model_path(name: str) -> str:
    return os.path.join(os.getcwd(), MODEL_PATHS[name])


def models_exist() -> bool:
    required_files = [MODEL_PATHS["rf"], MODEL_PATHS["scaler"], MODEL_PATHS["lr"]]
    return all(os.path.exists(path) for path in required_files)


def load_models():
    if not models_exist():
        raise FileNotFoundError(
            "Modelos não encontrados. Treine-os primeiro executando handleLearning() ou chamando o endpoint /model."
        )

    global modelo_rf, scaler, modelo_lr
    modelo_rf = joblib.load(_resolve_model_path("rf"))
    scaler = joblib.load(_resolve_model_path("scaler"))
    modelo_lr = joblib.load(_resolve_model_path("lr"))


try:
    load_models()
except Exception as exc:
    logger.warning(f"⚠️ Não foi possível carregar os modelos no momento: {exc}")
    modelo_rf = None
    scaler = None
    modelo_lr = None

def handleLearning():
    logger.info("🚀 [START] Iniciando o pipeline completo de treinamento (/model)...")
    inicio_pipeline = time.time()
    
    # 1. Leitura e tratamento dos dados[cite: 5]
    df = pd.read_csv('data.csv')
    
    cols_validar = ['area_mean', 'compactness_mean', 'perimeter_mean', 'concavity_mean', 'radius_mean']
    for col in cols_validar:
        mediana = df[col].median()
        df.loc[df[col] <= 0, col] = mediana
    
    df.drop_duplicates(inplace=True)

    X_completo = df[cols_validar]
    
    # Suporte caso o diagnosis já venha numérico do endpoint /confirm ou texto 'M'/'B'[cite: 5]
    if df['diagnosis'].dtype == object:
        y_completo = df['diagnosis'].map({'M': 1, 'B': 0})
    else:
        y_completo = df['diagnosis']
    
    X_train, X_test, y_train, y_test = train_test_split(X_completo, y_completo, test_size=0.2, random_state=42)   
     
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) 
    X_test_scaled = scaler.transform(X_test) 
    
    logger.info(f"📊 Dados preparados com sucesso. Amostras de Treino: {len(X_train_scaled)} | Amostras de Teste: {len(X_test_scaled)}")

    # =========================================================================
    # NOVO: MODELO BASELINE ORIGINAL (REQUISITO EXIGIDO NO PDF)[cite: 2, 5]
    # =========================================================================
    logger.info("📐 Avaliando o Modelo Original (Random Forest Padrão do Scikit-Learn)...")
    rfc_baseline = RandomForestClassifier(random_state=42)
    rfc_baseline.fit(X_train_scaled, y_train)
    y_pred_baseline = rfc_baseline.predict(X_test_scaled)
    f1_baseline = f1_score(y_test, y_pred_baseline, average='binary')
    logger.info(f"📉 F1-Score do Modelo Original (Baseline): {f1_baseline:.4f}")
    # =========================================================================

    # 3. EXPERIMENTOS DO ALGORITMO GENÉTICO[cite: 5]
    logger.info("⚡ Iniciando a fase de otimização via Algoritmos Genéticos...")
    otimizador = GeneticOptimizer(X_train_scaled, y_train)
    
    # Experimento 1[cite: 5]
    logger.info("⚙️ Rodando Experimento 1/3...")
    params_exp1, f1_exp1 = otimizador.rodar_experimento(num_geracoes=3, tam_populacao=6, taxa_mutacao=0.1)
    
    # Experimento 2[cite: 5]
    logger.info("⚙️ Rodando Experimento 2/3...")
    params_exp2, f1_exp2 = otimizador.rodar_experimento(num_geracoes=3, tam_populacao=10, taxa_mutacao=0.3)
    
    # Experimento 3[cite: 5]
    logger.info("⚙️ Rodando Experimento 3/3...")
    params_exp3, f1_exp3 = otimizador.rodar_experimento(num_geracoes=5, tam_populacao=8, taxa_mutacao=0.2)
    
    resultados = {f1_exp1: params_exp1, f1_exp2: params_exp2, f1_exp3: params_exp3}
    melhor_f1_encontrado = max(resultados.keys())
    genes_campeoes = resultados[melhor_f1_encontrado]
    
    best_n_estimators = int(genes_campeoes[0])
    best_max_depth = int(genes_campeoes[1])
    best_min_samples_split = int(genes_campeoes[2])
    
    logger.info(f"🏆 Otimização Concluída! Hiperparâmetros Vencedores -> n_estimators: {best_n_estimators}, max_depth: {best_max_depth}, min_samples_split: {best_min_samples_split}")

    # 4. TREINAMENTO DEFINITIVO COM OS PARÂMETROS OTIMIZADOS[cite: 5]
    logger.info("🏋️ Treinando os modelos finais com os hiperparâmetros otimizados...")
    
    rfc = RandomForestClassifier(
        n_estimators=best_n_estimators, 
        max_depth=best_max_depth, 
        min_samples_split=best_min_samples_split, 
        random_state=42
    )
    rfc.fit(X_train_scaled, y_train)
    
    lr_model = LogisticRegression()
    lr_model.fit(X_train_scaled, y_train)
    
    # 5. VALIDAÇÃO[cite: 5]
    y_pred = rfc.predict(X_test_scaled)
    y_pred_lr = lr_model.predict(X_test_scaled)
    f1_otimizado = f1_score(y_test, y_pred, average='binary')
    
    print("\n=== RELATÓRIO DE CLASSIFICAÇÃO RANDOM FOREST (OTIMIZADO AM) ===")
    print(classification_report(y_test, y_pred))

    # =========================================================================
    # NOVO: LOG COMPARATIVO EXIGIDO PELO RELATÓRIO DO TECH CHALLENGE[cite: 2, 5]
    # =========================================================================
    ganho_f1 = f1_otimizado - f1_baseline
    logger.info(f"📊 [COMPARAÇÃO OBRIGATÓRIA] F1 Baseline: {f1_baseline:.4f} vs F1 Otimizado: {f1_otimizado:.4f} | Ganho Líquido: {ganho_f1:+.4f}")
    
    # Matrizes de confusão e gráficos[cite: 5]
    plt.figure(figsize=(5,4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão - Random Forest Otimizado')
    plt.savefig('matriz_rf.png')

    plt.figure(figsize=(5,4))
    sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap='Greens')
    plt.title('Matriz de Confusão - Regressão Logística')
    plt.savefig('matriz_lr.png') 
    
    # 6. EXPORTAÇÃO[cite: 5]
    joblib.dump(rfc, 'trained_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(lr_model, 'logistic_model.pkl')
    
    importances = rfc.feature_importances_
    feature_names = ['area_mean', 'compactness_mean', 'perimeter_mean', 'concavity_mean', 'radius_mean']
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
    plt.title('Explicabilidade: Importância das Características (RF Otimizado)')
    plt.savefig('feature_importance.png')
    
    tempo_total_pipeline = time.time() - inicio_pipeline
    logger.info(f"🏁 [SUCCESS] Pipeline de treinamento completo finalizado com sucesso em {tempo_total_pipeline:.2f}s!")
    
def handlePrediction(patient_data: PatientDiagnosticModel):
    if not models_exist():
        raise FileNotFoundError(
            "Modelos não encontrados. Gere-os primeiro executando handleLearning() ou chamando o endpoint /model."
        )

    if modelo_rf is None or scaler is None or modelo_lr is None:
        load_models()

    features = ['area_mean', 'compactness_mean', 'perimeter_mean', 'concavity_mean', 'radius_mean']
    
    data_df = pd.DataFrame([[
        patient_data.area_mean,
        patient_data.compactness_mean, 
        patient_data.perimeter_mean,
        patient_data.concavity_mean,
        patient_data.radius_mean
    ]], columns=features)
    
    scaledData = scaler.transform(data_df)
        
    prob_rf = modelo_rf.predict_proba(scaledData)[0][1] * 100
    pred_rf = modelo_rf.predict(scaledData)[0]
    
    prob_lr = modelo_lr.predict_proba(scaledData)[0][1] * 100
    pred_lr = modelo_lr.predict(scaledData)[0]
    
    consensus = 2
    if pred_rf == 1 and pred_lr == 1: consensus = 1
    if pred_rf == 0 and pred_lr == 0: consensus = 0
    
    coeficientes = pd.DataFrame(modelo_lr.coef_, columns=['area_mean', 'compactness_mean', 'perimeter_mean', 'concavity_mean', 'radius_mean'])
    print("Pesos da Regressão Logística:\n", coeficientes)

    return {
        "random_forest": {"prediction": int(pred_rf), "risk": round(prob_rf, 2)},
        "logistic_regression": {"prediction": int(pred_lr), "risk": round(prob_lr, 2)},
        "final_consensus": consensus
    }