"""Fachada do assistente: gerencia sessões e normaliza a resposta do workflow."""

from __future__ import annotations

import json
import uuid
from collections import defaultdict, deque
from typing import Any

from pydantic import BaseModel, Field

from observability.logger import logger, mask_identifier
from langgraph_workflows.assistant_graph import DISCLAIMER, run_assistant_graph

#Essa classe recebe a resposta gerada pelo grafo. Pega o estado atual
#armazena em sessions para contextualizar e retorna.

_sessions: dict[str, deque[dict[str, str]]] = defaultdict(lambda: deque(maxlen=12))
# Mantém somente as últimas mensagens para fornecer contexto sem crescer indefinidamente.


class AssistantResponse(BaseModel):
    response: str
    sources: list[str] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER
    session_id: str
    source_details: list[dict[str, Any]] = Field(default_factory=list)
    context_used: dict[str, bool] = Field(default_factory=dict)
    correlation_id: str = ""


def _parse_tool_result(value: str) -> dict[str, Any]:
    # As ferramentas retornam JSON, mas o assistente continua resiliente a respostas malformadas.
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {"warning": "A ferramenta retornou dados inválidos.", "sources": [], "raw": value}


def _sanitize_response_final(text: str) -> str:
    """Última camada de sanitização antes de retornar ao usuário."""
    import re
    text = str(text)
    # Remove qualquer menção de arquivo .md, .csv etc
    text = re.sub(r'(?i)arquivo.*?\.md', '', text)
    text = re.sub(r'(?i)protocolo.*?\.md', '', text)
    text = re.sub(r'\b[a-zA-Z_]+\.(md|csv|json|txt)\b', '', text)
    # Remove "seção/parágrafo" ou similar
    text = re.sub(r'\(seção/parágrafo\s+\d+\)', '', text)
    # Remove paths
    text = re.sub(r'[a-zA-Z0-9_\-/]+/[a-zA-Z0-9_\-/]+\.(md|csv|txt|json)', '', text)
    # Limpa espaços múltiplos e quebras de linha
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


#recebe parametros da requisicao, remove espacos em branco
#e envia para o grafo.
def run_assistant(patient_id: str, query: str, session_id: str) -> AssistantResponse:
    patient_id = str(patient_id).strip()
    query = str(query).strip()
    session_id = str(session_id).strip() or "default"
    if not query:
        raise ValueError("query não pode ser vazia.")

    correlation_id = str(uuid.uuid4())

    try:
        # O grafo concentra guardrails, ferramentas e geração; esta camada apenas prepara o estado.
        state = run_assistant_graph({
            "correlation_id": correlation_id,
            "patient_id": patient_id,
            "query": query,
            "session_id": session_id,
            "previous_messages": list(_sessions[session_id]),
        })
    except Exception:
        logger.exception("assistant_chat_failed correlation_id=%s", correlation_id)
        raise

    response_text = state["response"]
    # Aplicar sanitização final
    response_text = _sanitize_response_final(response_text)
    
    _sessions[session_id].append({"query": query, "response": response_text})
    logger.info(
        "assistant_chat correlation_id=%s session_id=%s patient_id=%s provider=%s sources=%s guardrail_blocked=%s",
        correlation_id,
        session_id,
        mask_identifier(patient_id),
        state.get("model_provider", "unknown"),
        state.get("sources", []),
        state.get("guardrail_blocked", False),
    )
    return AssistantResponse(
        response=response_text,
        sources=state.get("sources", []),
        session_id=session_id,
        source_details=state.get("source_details", []),
        context_used={
            "patient_history": bool(state.get("history", {}).get("records")),
            "medical_protocols": bool(state.get("protocols", {}).get("results")),
            "synthetic_history": any(item.get("synthetic") for item in state.get("history", {}).get("source_details", [])),
        },
        correlation_id=correlation_id,
    )