import json
import random
from pathlib import Path


def build_risk_response(label: str, score: float, circ: float, area: float) -> str:
    """Gera resposta alinhada ao PCDT para risco alto, moderado e sem risco aparente."""
    if label == "sem risco aparente":
        return (
            "Para lesão classificada como sem risco aparente, a abordagem recomendada é manter a avaliação clínica e a imagem em rotina, "
            "sem impulsionar investigação invasiva apenas pelo valor numérico isolado. O score de {score}% e a área de {area} mm² devem ser correlacionados com o exame físico e a mamografia/ultrassonografia. "
            "Se o achado for estável e clínicamente benigno, segue-se acompanhamento conforme protocolo, com validação médica obrigatória."
        ).format(score=score, area=area)

    if label == "risco moderado":
        return (
            "Para lesão com perfil de risco moderado, recomenda-se avaliação clínica completa, correlação com imagem (mamografia e ultrassonografia complementares) e acompanhamento rigoroso conforme achados e estadiamento. "
            "A área de {area} mm² e o score de {score}% elevam a atenção, mas a conduta definitiva depende da concordância entre exame físico, imagem e contexto clínico. "
            "Em caso de suspeita persistente, a investigação pode avançar para biópsia dirigida e avaliação multidisciplinar, sempre com validação médica."
        ).format(score=score, area=area)

    return (
        "Para lesão classificada como risco elevado (score {score}%, circularidade {circ}, área {area} mm²), a condução correta é: 1) avaliação clínica completa; 2) correlação com mamografia e ultrassonografia complementar; 3) investigação diagnóstica dirigida com biópsia quando houver suspeita radiológica ou clínica; 4) estadiamento e discussão multidisciplinar quando a malignidade for confirmada. "
        "O valor numérico isolado não substitui o diagnóstico definitivo; a decisão terapêutica deve ser baseada na combinação de exame físico, imagem e anatomopatológico. Validação médica obrigatória antes de qualquer conduta."
    ).format(score=score, circ=circ, area=area)


def generate_expanded_dataset():
    output_path = Path("fine_tunning/dataset_anonimizado.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    risks = [
        ("sem risco aparente", 10.0, 30.0, 800.0, 950.0),
        ("risco moderado", 30.1, 65.0, 1000.0, 1450.0),
        ("risco elevado", 65.1, 98.0, 1500.0, 2300.0),
    ]

    templates = [
        "Qual é a conduta recomendada para uma paciente com lesão classificada como {label} com score de {score}% e circularidade de {circ}?",
        "De acordo com o protocolo, como proceder para avaliação de exames em lesão com classificação {label}, pontuação de {score}% e área de {area} mm²?",
        "Qual o próximo passo na investigação para um achado de {label} (risco de {score}%, circularidade {circ})?",
    ]

    dataset = []
    for _ in range(200):
        label, min_s, max_s, min_a, max_a = random.choice(risks)
        score = round(random.uniform(min_s, max_s), 1)
        circ = round(random.uniform(0.50, 0.95), 2)
        area = round(random.uniform(min_a, max_a), 1)

        prompt_template = random.choice(templates)
        user_content = prompt_template.format(label=label, score=score, circ=circ, area=area)
        assistant_content = build_risk_response(label, score, circ, area)

        entry = {
            "messages": [
                {
                    "role": "system",
                    "content": "Você é um assistente médico especialista em oncologia mamária e radiologia, treinado com os protocolos internos do hospital e a Portaria Conjunta nº 17 do PCDT de Câncer de Mama. Forneça análises baseadas em evidências, sempre ressaltando a necessidade de validação médica humana. Nunca prescreva medicamentos ou procedimentos definitivos. Cite sempre as fontes das diretrizes utilizadas.",
                },
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
            "metadata": {
                "risk_score": score,
                "risk_label": label,
                "area_mean": area,
                "circularity": circ,
                "patient_hash": f"PACIENTE_HASH_{random.randint(1000, 9999)}",
            },
        }
        dataset.append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✓ Dataset gerado com sucesso!")
    print(f"  Total de exemplos: {len(dataset)}")
    print(f"  Localização: {output_path}")
    print(f"  Qualidade: respostas alinhadas ao protocolo e sem frases artificiais de neoadjuvância.")


if __name__ == "__main__":
    generate_expanded_dataset()