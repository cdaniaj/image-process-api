from __future__ import annotations

import json
import time
from typing import Any, TypedDict

from langchain_agent.providers import generate_assistant_response
from langchain_agent.tools import search_medical_protocols, search_patient_history
from observability.logger import logger, mask_identifier

#Define os passos e ordem dos passos que devem ser tomadas pela IA para
#processar uma requisicao de ponta a ponta
#analogia: É como uma esteira de uma cafeteria. Tem todo um passo a passo
#para produzir um flat-white, e no final, o guardrails nada mais é que a
#vigilancia sanitaria garantindo seguranca e qualidade.

DISCLAIMER = "Sugestão de IA para auxílio médico. Validação humana obrigatória."
SYSTEM_PROMPT = (
    "Você é um assistente de apoio à decisão clínica. Não faça diagnóstico definitivo, "
    "não prescreva medicamentos ou tratamentos e não substitua um profissional habilitado. "
    "Use somente os contextos fornecidos, declare quando houver dados insuficientes e cite as fontes."
)
# Lista simples de padrões (substring match); não substitui um classificador de segurança real,
# mas amplia a cobertura contra prescrição/diagnóstico direto e paráfrases comuns.
FORBIDDEN_PATTERNS = (
    "prescrevo", "prescrição:", "tome ", "tomar ", "administre", "administrar",
    "inicie o tratamento", "iniciar tratamento", "aplique a dose", "dose recomendada:",
    "posologia", "receite", "diagnóstico confirmado", "diagnóstico definitivo",
    "você tem câncer", "é maligno", "é benigno", "confirmo o diagnóstico",
    "recomendo a cirurgia", "indico a cirurgia", "use o medicamento", "utilize o medicamento",
)
# Termos que, se presentes na resposta, já evidenciam citação de fonte/protocolo (evita duplicidade).
_SOURCE_MENTION_TERMS = ("fonte", "protocolo", "histórico", "diretriz", "pcdt", "portaria")


class AssistantState(TypedDict, total=False):
    correlation_id: str
    patient_id: str
    query: str
    session_id: str
    previous_messages: list[dict[str, str]]
    history: dict[str, Any]
    protocols: dict[str, Any]
    prompt: str
    response: str
    sources: list[str]
    source_details: list[dict[str, Any]]
    model_provider: str
    guardrail_blocked: bool


def _invoke_tool(tool: Any, value: str) -> dict[str, Any]:
    result = tool.invoke(value) if hasattr(tool, "invoke") else tool(value)
    try:
        return json.loads(result)
    except json.JSONDecodeError:
        return {"records": [], "results": [], "sources": [], "warning": "Resposta inválida da ferramenta."}


def validate_input(state: AssistantState) -> AssistantState:
    if not state.get("query", "").strip():
        raise ValueError("query não pode ser vazia.")
    return state


def retrieve_history(state: AssistantState) -> AssistantState:
    state["history"] = _invoke_tool(search_patient_history, state.get("patient_id", ""))
    logger.info(
        "assistant_graph_node correlation_id=%s node=retrieve_history patient_id=%s record_count=%s",
        state.get("correlation_id", "unknown"),
        mask_identifier(state.get("patient_id", "")),
        state["history"].get("record_count", len(state["history"].get("records", []))),
    )
    return state


def retrieve_protocols(state: AssistantState) -> AssistantState:
    state["protocols"] = _invoke_tool(search_medical_protocols, state["query"])
    logger.info(
        "assistant_graph_node correlation_id=%s node=retrieve_protocols result_count=%s",
        state.get("correlation_id", "unknown"),
        len(state["protocols"].get("results", [])),
    )
    return state


def _sanitize_context(text: str) -> str:
    """Remove referências de arquivo e paths do contexto antes de enviar ao modelo."""
    import re
    text = str(text)
    # Remove paths completos
    text = re.sub(r'image-process-api[/\\]', '', text)
    text = re.sub(r'[a-zA-Z_]+\.md', '', text)
    text = re.sub(r'[a-zA-Z_]+\.csv', '', text)
    # Remove "seção/parágrafo" que pode vazar
    text = re.sub(r'\s*\(seção/parágrafo\s+\d+\)', '', text)
    return text


def build_prompt(state: AssistantState) -> AssistantState:
    history = state.get("history", {})
    protocols = state.get("protocols", {})
    state["sources"] = list(dict.fromkeys(history.get("sources", []) + protocols.get("sources", [])))
    state["source_details"] = history.get("source_details", []) + protocols.get("source_details", [])
    
    # Sanitiza o contexto antes de construir o prompt
    history_json = _sanitize_context(json.dumps(history, ensure_ascii=False))
    protocols_json = _sanitize_context(json.dumps(protocols, ensure_ascii=False))
    
    state["prompt"] = (
        f"{SYSTEM_PROMPT}\n\n"
        f"ID do paciente: {state.get('patient_id', 'não informado')}\n"
        f"Pergunta do usuário: {state['query']}\n"
        f"---\n"
        f"HISTÓRICO DO PACIENTE:\n{history_json}\n"
        f"---\n"
        f"DIRETRIZES E PROTOCOLOS:\n{protocols_json}\n"
        f"---\n"
        f"Conversa anterior:\n{json.dumps(state.get('previous_messages', []), ensure_ascii=False)}\n"
        f"---\n"
        f"INSTRUÇÕES FINAIS:\n"
        f"1. Responda de forma objetiva e profissional\n"
        f"2. Mencione limitações de dados se houver\n"
        f"3. Cite as fontes de informação (protocolos, histórico do paciente)\n"
        f"4. Nunca referencie nomes de arquivos ou caminhos\n"
        f"5. Sempre mantenha o tom clínico apropriado\n"
        f"6. Nunca substitua avaliação médica profissional"
    )
    return state


def generate_response(state: AssistantState) -> AssistantState:
    response, provider = generate_assistant_response(state["prompt"])
    state["model_provider"] = provider
    if response:
        state["response"] = response
    else:
        history = state.get("history", {})
        protocols = state.get("protocols", {})
        history_summary = (
            f"Foram encontrados {len(history.get('records', []))} registro(s) para o paciente informado."
            if history.get("records")
            else history.get("warning", "Não foi possível localizar histórico.")
        )
        protocol_summary = (
            f"Há {len(protocols.get('results', []))} trecho(s) de protocolo relevante(s)."
            if protocols.get("results")
            else protocols.get("warning", "Não há protocolos disponíveis para consulta.")
        )
        state["response"] = f"{history_summary} {protocol_summary} Pergunta recebida: {state['query']}. A revisão e a conduta devem ser feitas por profissional habilitado."
    logger.info(
        "assistant_graph_node correlation_id=%s node=generate_response provider=%s",
        state.get("correlation_id", "unknown"),
        provider,
    )
    return state


def enforce_explainability(state: AssistantState) -> AssistantState:
    """Garante que toda resposta cite explicitamente a fonte/protocolo usado (requisito de explainability)."""
    if state.get("guardrail_blocked"):
        return state

    response = state.get("response", "")
    sources = state.get("sources", [])
    already_cites_source = any(term in response.lower() for term in _SOURCE_MENTION_TERMS)

    if already_cites_source:
        return state

    if sources:
        state["response"] = f"{response}\n\nFontes consultadas: {', '.join(sources)}."
    else:
        state["response"] = (
            f"{response}\n\nNenhuma fonte de protocolo ou histórico foi localizada para embasar esta resposta; "
            "trata-se de uma limitação do contexto disponível."
        )
    return state


def validate_guardrails(state: AssistantState) -> AssistantState:
    response = state.get("response", "")
    blocked = any(pattern in response.lower() for pattern in FORBIDDEN_PATTERNS)
    state["guardrail_blocked"] = blocked
    if blocked:
        state["response"] = "Não posso fornecer prescrição ou diagnóstico definitivo. Posso organizar os dados e as fontes para revisão por profissional habilitado."
    logger.info(
        "assistant_graph_node correlation_id=%s node=validate_guardrails blocked=%s",
        state.get("correlation_id", "unknown"),
        blocked,
    )
    return state


#define os passos e a ordem de execucao da pipeline de IA
def _audited(name: str, func):
    """Envolve um nó do grafo com log de duração e correlation_id, sem expor dados sensíveis."""
    def wrapper(state: AssistantState) -> AssistantState:
        start = time.perf_counter()
        try:
            return func(state)
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.info(
                "assistant_graph_timing correlation_id=%s node=%s duration_ms=%s",
                state.get("correlation_id", "unknown"),
                name,
                duration_ms,
            )
    return wrapper


def _build_graph():
    from langgraph.graph import END, START, StateGraph

    nodes = {
        "validate_input": validate_input,
        "retrieve_history": retrieve_history,
        "retrieve_protocols": retrieve_protocols,
        "build_prompt": build_prompt,
        "generate_response": generate_response,
        "enforce_explainability": enforce_explainability,
        "validate_guardrails": validate_guardrails,
    }
    graph = StateGraph(AssistantState)
    for name, func in nodes.items():
        graph.add_node(name, _audited(name, func))
    graph.add_edge(START, "validate_input")
    graph.add_edge("validate_input", "retrieve_history")
    graph.add_edge("retrieve_history", "retrieve_protocols")
    graph.add_edge("retrieve_protocols", "build_prompt")
    graph.add_edge("build_prompt", "generate_response")
    graph.add_edge("generate_response", "enforce_explainability")
    graph.add_edge("enforce_explainability", "validate_guardrails")
    graph.add_edge("validate_guardrails", END)
    return graph.compile()


def run_assistant_graph(state: AssistantState) -> AssistantState:
    start = time.perf_counter()
    try:
        try:
            result = _build_graph().invoke(state)
        except ImportError:
            # Permite executar o backend sem LangGraph instalado em ambientes mínimos.
            result = state
            for node_name, node in (
                ("validate_input", validate_input),
                ("retrieve_history", retrieve_history),
                ("retrieve_protocols", retrieve_protocols),
                ("build_prompt", build_prompt),
                ("generate_response", generate_response),
                ("enforce_explainability", enforce_explainability),
                ("validate_guardrails", validate_guardrails),
            ):
                result = _audited(node_name, node)(result)
        return result
    finally:
        total_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "assistant_graph_total correlation_id=%s duration_ms=%s",
            state.get("correlation_id", "unknown"),
            total_ms,
        )