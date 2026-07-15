import random
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from observability.logger import logger 

class GeneticOptimizer:
    def __init__(self, X=None, y=None, data_path="data.csv"):
        self.data_path = data_path
        self.X = X
        self.y = y

        if self.X is None or self.y is None:
            self._load_data_from_csv()

    def _load_data_from_csv(self):
        df = pd.read_csv(self.data_path)
        features = ['area_mean', 'compactness_mean', 'perimeter_mean', 'concavity_mean', 'radius_mean']
        X = df[features]
        y = df['diagnosis']

        if y.dtype == object:
            y = y.map({'M': 1, 'B': 0})

        self.X = X.to_numpy()
        self.y = y.to_numpy()
        
    def calcular_fitness(self, individual):
        """
        Mapeia os genes para os hiperparâmetros do Random Forest e calcula o F1-Score[cite: 27, 29].
        Individual: [n_estimators, max_depth, min_samples_split]
        """
        n_estimators = int(individual[0])
        max_depth = int(individual[1])
        min_samples_split = int(individual[2])
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        
        # Executa cross-validation usando F1-Score como fitness 
        try:
            scores = cross_val_score(model, self.X, self.y, cv=3, scoring='f1')
            return scores.mean()
        except:
            return 0.0

    def rodar_experimento(self, num_geracoes, tam_populacao, taxa_mutacao):
        """
        Roda um experimento completo do AG com operadores de seleção, cruzamento e mutação[cite: 29, 31].
        """
        inicio = time.time()
        logger.info(f"🧬 Iniciando Experimento AG: Pop={tam_populacao}, Gerações={num_geracoes}, Mutação={taxa_mutacao} [cite: 31, 33]")
        
        # 1. Inicializa população aleatória
        # Cada indivíduo: [n_estimators (10-200), max_depth (2-20), min_samples_split (2-10)]
        populacao = [
            [random.randint(10, 200), random.randint(2, 20), random.randint(2, 10)]
            for _ in range(tam_populacao)
        ]
        
        for geracao in range(num_geracoes):
            inicio_geracao = time.time()
            
            # Avaliação (Fitness) 
            fitnesses = [self.calcular_fitness(ind) for ind in populacao]
            
            # Seleção (Ex: Torneio) 
            selecionados = [random.choice(populacao) for _ in range(tam_populacao)] 
            
            # Cruzamento (Crossover) 
            proxima_geracao = []
            for i in range(0, tam_populacao, 2):
                pai1, pai2 = selecionados[i], selecionados[i+1]
                if random.random() < 0.7:  # Taxa de Crossover
                    ponto = random.randint(1, 2)
                    filho1 = pai1[:ponto] + pai2[ponto:]
                    filho2 = pai2[:ponto] + pai1[ponto:]
                else:
                    filho1, filho2 = pai1.copy(), pai2.copy()
                proxima_geracao.extend([filho1, filho2])
                
            # Mutação 
            for ind in proxima_geracao:
                if random.random() < taxa_mutacao:
                    ind[0] = max(10, ind[0] + random.randint(-20, 20))
                    ind[1] = max(2, ind[1] + random.randint(-2, 2))
            
            populacao = proxima_geracao
            melhor_f1 = max(fitnesses)
            
            tempo_geracao = time.time() - inicio_geracao
            logger.info(f"⏱️ Geração {geracao+1}/{num_geracoes} concluída em {tempo_geracao:.2f}s | Melhor F1: {melhor_f1:.4f} ")
            
        tempo_total = time.time() - inicio
        logger.info(f"✅ Experimento Finalizado em {tempo_total:.2f}s ")
        
        # Retorna o melhor indivíduo do experimento final
        melhor_ind = populacao[fitnesses.index(max(fitnesses))]
        return melhor_ind, max(fitnesses)