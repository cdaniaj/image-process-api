import pytest

def calcular_ganho_f1(f1_baseline: float, f1_otimizado: float) -> float:
    """
    Função helper para replicar a lógica de cálculo usada no seu model.py.
    """
    return f1_otimizado - f1_baseline

def test_logic_performance_gain():
    """
    Valida se o cálculo do ganho de performance entre modelos está correto.
    """
    # Cenário 1: Ganho positivo (Otimizado melhor que Baseline)
    f1_base = 0.8500
    f1_opt = 0.9200
    ganho = calcular_ganho_f1(f1_base, f1_opt)
    assert ganho == pytest.approx(0.0700) # Usamos approx para lidar com floats

    # Cenário 2: Ganho negativo (Modelo piorou)
    f1_base = 0.9000
    f1_opt = 0.8800
    ganho = calcular_ganho_f1(f1_base, f1_opt)
    assert ganho == pytest.approx(-0.0200)

    # Cenário 3: Empate técnico
    f1_base = 0.9000
    f1_opt = 0.9000
    ganho = calcular_ganho_f1(f1_base, f1_opt)
    assert ganho == 0.0