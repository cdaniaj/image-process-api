import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def getReports():
    # 1. Carregar os dados
    df = pd.read_csv('data.csv')

    # Selecionar apenas as colunas que você está usando no projeto
    features = ['area_mean', 'compactness_mean', 'perimeter_mean', 'concavity_mean', 'radius_mean']
    target = 'diagnosis'

    # --- ANÁLISE DE CORRELAÇÃO ---
    plt.figure(figsize=(10, 8))
    # Calculamos a correlação apenas das features numéricas
    corr = df[features].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matriz de Correlação das Características Selecionadas')
    plt.show()

    # --- DISTRIBUIÇÃO DAS FEATURES POR DIAGNÓSTICO ---
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
    axes = axes.flatten()

    for i, col in enumerate(features):
        sns.histplot(data=df, x=col, hue=target, kde=True, ax=axes[i], palette='magma')
        axes[i].set_title(f'Distribuição de {col}')

    # Remove o último subplot vazio (já que temos 5 features e 6 espaços)
    fig.delaxes(axes[5])
    plt.tight_layout()
    plt.show()

    # --- BOXPLOT PARA IDENTIFICAÇÃO DE OUTLIERS ---
    plt.figure(figsize=(15, 6))
    sns.boxplot(data=df[features], palette='Set3')
    plt.title('Identificação de Outliers nas Métricas Selecionadas')
    plt.show()