# image-process-api/llm_layer/client.py
import os
import google.generativeai as genai
from fastapi import HTTPException

# Configura a chave de API (recomendo puxar do ambiente/arquivo .env)
GEMINI_API_KEY = "AQ.Ab8RN6J5AAyHbCqDhtaJ5Hx8c9I_jC-1mBYPhguB8ukIjVcsXw"
if not GEMINI_API_KEY:
    # Levanta um aviso ou erro caso esqueça de configurar a chave
    raise ValueError("A variável de ambiente GEMINI_API_KEY não foi configurada.")

genai.configure(api_key=GEMINI_API_KEY)
    
def generate_medical_report(patient_data: dict) -> str:
    """
    Recebe os dados extraídos e o score de risco da API e envia para o 
    Gemini 2.5 flash gerar o laudo humanizado para o médico.
    """
    try:
        # Instancia o Gemini 1.5 Pro (modelo disponível na API)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Engenharia de Prompt focada no contexto hospitalar
        prompt = f"""
        Você é um assistente de IA especialista em oncologia mamária e radiologia.
        Sua tarefa é traduzir as métricas geométricas e estatísticas de uma lesão mamográfica em um laudo clínico descritivo, formal e acionável para o médico responsável.

        Dados Extraídos da Imagem:
        - Área Real/Média: {patient_data.get('area_mean')} mm²
        - Perímetro: {patient_data.get('perimeter_mean')} mm
        - Circularidade: {patient_data.get('circularity', 0)}
        - Solidez: {patient_data.get('solidity', 0)}

        Predição dos Modelos de ML:
        - Score de Risco Computado: {patient_data.get('risk_score')}%
        - Classificação de Risco: {patient_data.get('risk_label')}

        Com base nesses indicadores, escreva um relatório curto contendo:
        1. Análise clínica preliminar do aspecto da lesão (ex: se o formato irregular ou a solidez indicam maior atenção).
        2. Uma explicação textual sobre o score de risco apontado pela IA.
        3. Próximos passos sugeridos (ex: exames complementares, biópsia ou apenas acompanhamento de rotina).
        
        Seja direto, profissional e evite jargões desnecessários fora do escopo médico.
        """
        
        # Configuração para evitar alucinações (temperatura baixa = mais factual)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
            )
        )
        
        return response.text
        
    except Exception as e:
        # Tratamento de erro robusto como seu SDD preza
        raise HTTPException(status_code=500, detail=f"Erro na integração com a LLM: {str(e)}")