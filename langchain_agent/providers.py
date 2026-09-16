"""Providers de geração: Gemini remoto e adaptador PEFT local com fallback controlado."""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any
import torch


logger = logging.getLogger("image-process-api")

# Força o uso de uma única thread no PyTorch para evitar falhas de segmentação (SIGSEGV/EXC_BAD_ACCESS) no macOS/OpenMP
torch.set_num_threads(1)


def _generate_with_gemini(prompt: str) -> str:
    # O import tardio permite iniciar a API mesmo quando o provider opcional não está instalado.
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY não configurada para o provider Gemini.")

    model = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        google_api_key=api_key,
        temperature=0.2,
    )
    result = model.invoke(prompt)
    return result.content if isinstance(result.content, str) else str(result.content)


def _try_gemini_safely(prompt: str) -> str | None:
    """Tenta gerar com Gemini sem propagar exceções (sem chave, rede fora, etc.)."""
    if os.getenv("ASSISTANT_OFFLINE", "false").strip().lower() == "true":
        return None
    if not os.getenv("GEMINI_API_KEY", "").strip():
        return None
    try:
        return _generate_with_gemini(prompt)
    except Exception as exc:
        logger.warning("Gemini indisponível; usando fallback controlado: %s", exc)
        return None


@lru_cache(maxsize=1)
def _load_local_model():
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer
    from fine_tunning.train import resolve_adapter_path

    adapter_path = resolve_adapter_path()
    config = json.loads((adapter_path / "adapter_config.json").read_text(encoding="utf-8"))
    base_model = config.get("base_model_name_or_path")
    if not base_model:
        raise ValueError("adapter_config.json não informa base_model_name_or_path.")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Força torch.float32 para evitar NaNs e infs comuns no Phi-2 com float16 no Mac
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(adapter_path),
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def _sanitize_response(text: str) -> str:
    """Remove caracteres especiais, tags, e referências de arquivos com sanitização agressiva."""
    import re
    
    text = str(text)
    
    # 1. Remove todos os tokens especiais e tags conhecidas
    special_tokens = [
        "<|user|>", "<|system|>", "<|assistant|>", "<|end|>", 
        "<|endoftext|>", "<END>", "[END]", "]]", "[[",
        "<pad>", "<unk>", "[PAD]", "[UNK]"
    ]
    for token in special_tokens:
        text = text.replace(token, " ")
    
    # 2. Remove sequências problemáticas (números colados com letras de forma errada)
    # Padrão: número seguido imediatamente de letra (sem espaço) = erro
    text = re.sub(r'(\d+)([a-zà-ÿ])', r'\1 \2', text, flags=re.IGNORECASE)
    
    # 3. Remove palavras claramente fabricadas/sem sentido
    gibberish_patterns = [
        (r'\b[a-z]*[aeio]{3,}[a-z]*\b', ''),  # Palavras com muitos acentos
        (r'\b[a-z]ch[a-z]{2,}\b', ''),  # "dechocimentosa" pattern
        (r'\b[a-z]{2}[ç][a-z]{2,}\b', ''),  # "discussão" mal formatada
        (r'\b[a-z]{1,2}ção\b', ''),  # Terminações erradas
        (r'\b[a-z]*[aeiou]{4,}[a-z]*\b', ''),  # Muitas vogais seguidas
    ]
    
    for pattern, replacement in gibberish_patterns:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # 4. Remove caracteres de escape e malformações
    escape_patterns = [
        (r'\\[\"nrt\\]', " "),  # Escape sequences
        (r'["\]\}\{]+', " "),  # Chaves e aspas isoladas
        (r'(?i)arquivo|file|path|diretório|directory|\.md|\.csv|\.json|\.txt', ""),
        (r'\s+', ' '),  # Normaliza espaços
    ]
    
    for pattern, replacement in escape_patterns:
        text = re.sub(pattern, replacement, text)
    
    # 5. Remove textos que parecem corrompidos
    text = re.sub(r'[a-z]+[A-Z][a-z]+', lambda m: m.group(0).lower(), text)  # camelCase → minúsculas
    
    # 6. Normaliza pontuação
    text = re.sub(r'([.!?])\s*([a-z])', r'\1 \2', text)
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    
    # 7. Remove linhas vazias ou muito curtas (< 5 caracteres)
    lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 4]
    text = '\n'.join(lines)
    
    # 8. Normaliza quebras de linha
    text = re.sub(r'[\r\n]{2,}', '\n\n', text)
    
    # 9. Remove espaçamento no início/fim
    text = text.strip()
    
    # 10. Validação final: se texto ficou muito curto ou vazio
    if len(text) < 20:
        return ""
    
    return text


# Marcadores que indicam que o modelo "vazou" para código, bibliografia acadêmica
# fake ou outro texto de pré-treino não relacionado ao domínio clínico.
_CODE_MARKERS = (
    "```", "def ", "import ", "class ", "print(", "sys.exit", "return ",
    "except ", "elif ", "lambda ", "enumerate(", "f.readline", ".strip(",
    ".split(", "open(", "valueerror", "**solution", "**ideas", "exercise",
    "for line in", "with open", "# ", "->", "self.",
)

# Marcadores de "dump" bibliográfico/acadêmico alucinado (DOI, Scopus, URLs, metadados de citação).
# Esse é um padrão recorrente de alucinação do Phi-2 base quando o LoRA não segura o domínio.
_CITATION_DUMP_MARKERS = (
    "http://", "https://", "www.", "doi:", "doi.org", "issn", "urn:", "scopus",
    "electronic-address", "resolveurl", "linkinghub", "elsevier", "footnotes",
    "conclusions:", "evidence:", "volume=", "issue=", "spage=", "pages=",
    "@scopus", "ots=", "site=scopus",
)

# Palavras/expressões comuns em inglês que não deveriam aparecer numa resposta clínica em PT-BR
_ENGLISH_INDICATORS = (
    " the ", " and ", " should ", " using ", " each ", " based on ",
    " create a ", " read the ", " store ", " loop ", " print out ",
    " column", " key names", " validate", " threshold", " dictionary",
    " script", " input ", " output ",
)


def _looks_off_topic(text: str) -> bool:
    """Detecta vazamento para código, dump bibliográfico/citações fake ou texto genérico
    em inglês — falhas de geração conhecidas do Phi-2 fine-tuned nesse domínio."""
    import re

    lowered = f" {text.lower()} "

    if any(marker in lowered for marker in _CODE_MARKERS):
        return True

    if any(marker in lowered for marker in _CITATION_DUMP_MARKERS):
        return True

    # Estrutura de "chave: valor" repetida (metadados/bibliografia), atípica de prosa clínica.
    if len(re.findall(r'\b[a-zçãõáéíóú-]+:\s', lowered)) >= 4:
        return True

    # Excesso de colchetes/pipes indica dump estruturado em vez de texto corrido.
    if text.count("[") + text.count("]") >= 2 or text.count("|") >= 2:
        return True

    english_hits = sum(1 for marker in _ENGLISH_INDICATORS if marker in lowered)
    return english_hits >= 3


def _is_valid_response(text: str) -> tuple[bool, str]:
    """Verifica se a resposta é coerente, em português e pertence ao domínio médico."""
    import re
    
    text = str(text).strip()
    
    if _looks_off_topic(text):
        return False, "Conteúdo fora do domínio médico (código ou texto genérico em inglês detectado)."
    
    # Verificações
    checks = {
        "length": len(text) >= 50,  # Mínimo de conteúdo
        "has_period": "." in text,  # Tem pontuação básica
        "not_repetitive": not re.match(r'^(.{1,20})\1{3,}', text),  # Não é repetição
        "coherent_words": sum(1 for word in text.split() if len(word) > 2) >= 5,  # Mínimo de palavras reais
        "no_gibberish": not re.search(r'[aeiou]{5,}', text),  # Sem sequências anormais
        "portuguese_like": len(re.findall(r'[àáâãäæèéêëìíîïòóôõöœùúûüñçÀÁÂÃÄÆÈÉÊËÌÍÎÏÒÓÔÕÖŒÙÚÛÜÑÇ]', text)) > 0 or \
                          any(word in text.lower() for word in ['o', 'a', 'de', 'para', 'com', 'é', 'que', 'recomenda']),
    }
    
    valid = sum(checks.values()) >= 4  # Precisa passar em 4 de 6 verificações
    
    reason = "OK" if valid else f"Falhas: {[k for k, v in checks.items() if not v]}"
    
    return valid, reason


def _extract_clinical_params(prompt: str) -> dict[str, str]:
    """Extrai score, rótulo de risco, circularidade e área do prompt via regex, sem depender de IA."""
    import re

    label_match = re.search(
        r'(sem risco aparente|risco moderado|risco elevado|no_risk|moderate_risk|high_risk)',
        prompt,
        re.IGNORECASE,
    )
    score_match = re.search(r'score\s*(?:de|computado)?\s*:?\s*(\d+[.,]?\d*)\s*%', prompt, re.IGNORECASE)
    circ_match = re.search(r'circularidade\s*(?:de)?\s*:?\s*(\d+[.,]?\d*)', prompt, re.IGNORECASE)
    area_match = re.search(r'\u00e1rea\s*(?:de|real.{0,15})?\s*:?\s*(\d+[.,]?\d*)\s*mm', prompt, re.IGNORECASE)

    label_raw = (label_match.group(1) if label_match else "não especificado").lower()
    label_map = {
        "no_risk": "sem risco aparente",
        "moderate_risk": "risco moderado",
        "high_risk": "risco elevado",
    }
    label = label_map.get(label_raw, label_raw)

    return {
        "label": label,
        "score": score_match.group(1) if score_match else "não informado",
        "circularity": circ_match.group(1) if circ_match else "não informada",
        "area": area_match.group(1) if area_match else "não informada",
    }


def _extract_json_section(prompt: str, label: str) -> dict[str, Any]:
    """Extrai e desserializa o bloco JSON que o build_prompt embute após um rótulo de seção."""
    import re

    match = re.search(rf"{re.escape(label)}\n(.*?)\n---", prompt, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}


def _deterministic_clinical_fallback(prompt: str) -> str:
    """Resposta ancorada apenas em texto já recuperado pelo RAG (protocolos/histórico).

    Última linha de defesa: usada quando o modelo fine-tuned falha na validação
    de qualidade E o Gemini está indisponível (sem chave, offline ou erro de rede).
    Nunca gera texto livre — só recombina o que já foi de fato consultado, o que
    elimina alucinação por construção (não há geração de conteúdo novo).
    """
    params = _extract_clinical_params(prompt)
    protocols = _extract_json_section(prompt, "DIRETRIZES E PROTOCOLOS:")
    history = _extract_json_section(prompt, "HISTÓRICO DO PACIENTE:")

    parts = [
        f"Considerando os indicadores informados (classificação {params['label']}, score de {params['score']}%, "
        f"circularidade {params['circularity']} e área de {params['area']} mm²), recomenda-se avaliação clínica "
        "completa, ultrassonografia complementar e acompanhamento conforme estadiamento, com validação médica "
        "obrigatória antes de qualquer conduta."
    ]

    protocol_results = protocols.get("results") or []
    if protocol_results:
        top = protocol_results[0]
        excerpt = str(top.get("content", "")).strip()
        if len(excerpt) > 320:
            excerpt = excerpt[:320].rsplit(" ", 1)[0] + "..."
        source = str(top.get("source", "protocolo consultado"))
        if excerpt:
            parts.append(f'Trecho consultado em "{source}": "{excerpt}"')

    history_records = history.get("records") or []
    if history_records:
        parts.append(f"Foram localizados {len(history_records)} registro(s) no histórico do paciente para contextualizar esta resposta.")

    parts.append("Sugestão de IA para apoio à decisão; a conduta definitiva exige avaliação por profissional médico habilitado.")
    return " ".join(parts)


def _generate_with_fine_tuned(prompt: str) -> tuple[str, str]:
    """Gera resposta com modelo fine-tuned, com fallback automático para Gemini se ruim."""
    try:
        model, tokenizer = _load_local_model()
    except Exception:
        # Se não conseguir carregar o modelo, vai direto para Gemini (com proteção)
        gemini_response = _try_gemini_safely(prompt)
        if gemini_response is not None:
            return gemini_response, "gemini_fallback_load_error"
        return _deterministic_clinical_fallback(prompt), "template_fallback_load_error"
    
    tokenizer.truncation_side = "right"
    
    system_content = (
        "Você é um assistente médico especialista em oncologia mamária e radiologia, treinado com os protocolos internos do hospital e a Portaria Conjunta nº 17 do PCDT de Câncer de Mama. Forneça análises baseadas em evidências, sempre ressaltando a necessidade de validação médica humana. Cite sempre as fontes das diretrizes utilizadas."
    )
    
    # Recria a formatação idêntica ao training
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt}
    ]
    
    formatted_list = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        formatted_list.append(f"<|{role}|>\n{content}\n")
        
    formatted_prompt = "\n".join(formatted_list) + "\n<|assistant|>\n"
    
    inputs = tokenizer(
        formatted_prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=1536
    )
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Bane tokens que costumam indicar vazamento para código/inglês de pré-treino
    bad_words = ["```", " def ", " import ", " class ", " print(", " Exercise", " Solution", " sys.exit", " **Solution", " **Ideas"]
    bad_words_ids = [ids for ids in (tokenizer(word, add_special_tokens=False).input_ids for word in bad_words) if ids]
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=int(os.getenv("FINE_TUNING_MAX_NEW_TOKENS", "768")),
            min_new_tokens=80,  # Aumentado para forçar resposta mais longa
            do_sample=False,
            top_p=0.95,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,  # Evita repetir/continuar trechos memorizados do pré-treino
            bad_words_ids=bad_words_ids or None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True,
        )
    
    input_length = inputs["input_ids"].shape[1]
    generated = output[0][input_length:]
    response_text = tokenizer.decode(generated, skip_special_tokens=False).strip()
    
    # Sanitização
    response_text = _sanitize_response(response_text)
    
    # VERIFICAÇÃO DE QUALIDADE: se a resposta for ruim, usa Gemini com fallback seguro
    is_valid, reason = _is_valid_response(response_text)
    
    if not is_valid:
        gemini_response = _try_gemini_safely(prompt)
        if gemini_response is not None:
            return gemini_response, "gemini_fallback_quality"
        return _deterministic_clinical_fallback(prompt), "template_fallback_quality"
    
    return response_text, "fine_tuned"


def generate_assistant_response(prompt: str) -> tuple[str | None, str]:
    """Gera uma resposta usando o provider configurado.

    O provider padrão é Gemini. O provider `fine_tuned` exige o stack opcional
    e um adaptador PEFT válido em FINE_TUNING_OUTPUT_DIR.
    
    Retorna: (response_text, provider_name)
    """
    provider = os.getenv("ASSISTANT_MODEL_PROVIDER", "gemini").strip().lower()

    # ASSISTANT_OFFLINE precisa valer para qualquer provider: sem isso, testes e ambientes
    # sem GPU/rede acabam tentando baixar/carregar o modelo local e travam indefinidamente.
    if os.getenv("ASSISTANT_OFFLINE", "false").strip().lower() == "true":
        return None, "fallback"

    if provider == "fine_tuned":
        response_text, actual_provider = _generate_with_fine_tuned(prompt)
        return response_text, actual_provider
    
    if provider != "gemini":
        raise ValueError(f"ASSISTANT_MODEL_PROVIDER inválido: {provider}")
    
    if not os.getenv("GEMINI_API_KEY", "").strip():
        return None, "fallback"
    
    try:
        return _generate_with_gemini(prompt), "gemini"
    except Exception as exc:
        # Falhas de credencial ou rede não interrompem o fluxo: o grafo usa sua resposta determinística.
        logger.warning("Falha no provider Gemini; usando fallback controlado: %s", exc)
        return None, "fallback"