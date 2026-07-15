# image-process-api/llm_layer/client.py
import os
import re
import google.generativeai as genai
from fastapi import HTTPException

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if not GEMINI_API_KEY:
    raise ValueError("A variável de ambiente GEMINI_API_KEY não foi configurada.")

genai.configure(api_key=GEMINI_API_KEY)

#LLM AS A JUDGE
def evaluate_report(report_text: str) -> dict:
    """
    Avalia a qualidade do laudo gerado pela LLM usando critérios heurísticos.
    Requisito: Avaliação de qualidade das interpretações[cite: 2].
    """
    checks = {
        "menciona_area": bool(re.search(r'área|dimensões|mm²', report_text, re.IGNORECASE)),
        "menciona_risco": bool(re.search(r'risco|probabilidade|nível', report_text, re.IGNORECASE)),
        "menciona_conduta": bool(re.search(r'conduta|próximos passos|biópsia|acompanhamento', report_text, re.IGNORECASE)),
        "formato_adequado": len(report_text) > 150 # Garante que a LLM não respondeu algo vazio ou curto demais
    }
    
    score = sum(checks.values())
    return {
        "score_total": f"{score}/4",
        "valid": score >= 3,
        "detalhes": checks
    }
    
 #GENERATE MEDICAL REPORT USING PROMPT  
def generate_medical_report(patient_data: dict, clinical_notes: str = "") -> dict:
    """
    Gera o laudo e avalia sua qualidade.
    Adicionado suporte a clinical_notes para Módulo 3[cite: 2].
    """
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
        Você é um assistente de IA especialista em oncologia mamária e radiologia.
        Sua tarefa é traduzir as métricas geométricas e estatísticas de uma lesão mamográfica em um laudo clínico descritivo.

        Dados Extraídos da Imagem:
        - Área Real/Média: {patient_data.get('area_mean')} mm²
        - Perímetro: {patient_data.get('perimeter_mean')} mm
        - Circularidade: {patient_data.get('circularity', 0)}
        - Solidez: {patient_data.get('solidity', 0)}

        Predição dos Modelos de ML:
        - Score de Risco Computado: {patient_data.get('risk_score')}%
        - Classificação de Risco: {patient_data.get('risk_label')}
        
        Histórico Clínico (Contexto Módulo 3):
        {clinical_notes if clinical_notes else "Não fornecido."}

        Com base nesses indicadores, escreva um relatório curto contendo:
        1. Análise clínica preliminar.
        2. Explicação textual sobre o risco.
        3. Próximos passos sugeridos.
        
        Seja direto, profissional e evite jargões desnecessários.
        """
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.2)
        )
        
        report_text = response.text
        evaluation = evaluate_report(report_text)
        
        return {
            "report": report_text,
            "evaluation": evaluation
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na integração com a LLM: {str(e)}")