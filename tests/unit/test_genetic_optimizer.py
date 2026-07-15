import numpy as np

from ai_model.mamography_rf.genetic_optimizer import GeneticOptimizer


def test_calcular_fitness_returns_numeric_value():
    optimizer = GeneticOptimizer(np.array([[0.2, 0.3, 0.4]]), np.array([1]))
    fitness = optimizer.calcular_fitness([10, 5, 2])

    assert isinstance(fitness, float)
    assert np.isfinite(fitness)


def test_rodar_experimento_returns_two_values():
    optimizer = GeneticOptimizer(np.array([[0.2, 0.3, 0.4]]), np.array([1]))
    best_individual, best_fitness = optimizer.rodar_experimento(
        num_geracoes=1,
        tam_populacao=4,
        taxa_mutacao=0.0,
    )

    assert len(best_individual) == 3
    assert isinstance(best_fitness, float)
