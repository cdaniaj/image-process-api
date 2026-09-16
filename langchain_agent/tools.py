"""Ferramentas LangChain para consultar histórico de pacientes e protocolos locais."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from data_extraction.extraction import _read_csv_compat


#Tools personalizadas de RAG para embasar a IA generativa

try:
    from langchain_core.tools import tool
except ImportError:  # Permite executar o backend sem o stack opcional instalado.
    # A função original é suficiente para testes e execução degradada sem LangChain.
    def tool(func):
        return func


BASE_DIR = Path(__file__).resolve().parents[1]
CONFIRMED_DATASET = BASE_DIR / "data_extraction" / "gerados" / "dataset_extraction.csv"
LEGACY_DATASET = BASE_DIR / "data.csv"
SYNTHETIC_HISTORY_DATASET = Path(
    os.getenv("IMAGE_PROCESS_SYNTHETIC_HISTORY", BASE_DIR / "data" / "synthetic_patient_history.csv")
)
PROTOCOLS_DIR = Path(os.getenv("MEDICAL_PROTOCOLS_DIR", BASE_DIR / "protocols"))


def _dataset_frames() -> list[pd.DataFrame]:
    # Cada caminho é configurável para permitir trocar datasets sem alterar o código.
    frames = []
    for path in (
        Path(os.getenv("IMAGE_PROCESS_LEGACY_DATASET", LEGACY_DATASET)),
        Path(os.getenv("IMAGE_PROCESS_CONFIRMED_DATASET", CONFIRMED_DATASET)),
        SYNTHETIC_HISTORY_DATASET,
    ):
        if path.exists():
            if path.name == "dataset_extraction.csv":
                frames.append(_read_csv_compat(path))
            else:
                frames.append(pd.read_csv(path))
    return frames


@tool
def search_patient_history(patient_id: str) -> str:
    """Busca análises confirmadas e registros do paciente pelo identificador."""
    patient_id = str(patient_id).strip()
    if not patient_id:
        return json.dumps({"records": [], "sources": [], "warning": "patient_id não informado."}, ensure_ascii=False)

    records: list[dict[str, Any]] = []
    for frame in _dataset_frames():
        # Datasets antigos usam "id"; o formato atual usa "patient_id".
        id_column = "patient_id" if "patient_id" in frame.columns else "id" if "id" in frame.columns else None
        if id_column is None:
            continue
        matches = frame[frame[id_column].astype(str) == patient_id]
        records.extend(matches.where(matches.notna(), None).to_dict(orient="records"))

    records.sort(key=lambda record: str(record.get("exam_date", "")))

    source_details = []
    if Path(os.getenv("IMAGE_PROCESS_LEGACY_DATASET", LEGACY_DATASET)).exists():
        source_details.append({"name": "data.csv", "type": "legacy_training_dataset", "synthetic": False})
    if Path(os.getenv("IMAGE_PROCESS_CONFIRMED_DATASET", CONFIRMED_DATASET)).exists():
        source_details.append({"name": "dataset_extraction.csv", "type": "confirmed_analysis", "synthetic": False})
    if SYNTHETIC_HISTORY_DATASET.exists():
        source_details.append({"name": str(SYNTHETIC_HISTORY_DATASET.relative_to(BASE_DIR)), "type": "synthetic_history", "synthetic": True})
    result: dict[str, Any] = {
        "patient_id": patient_id,
        "records": records,
        "sources": [source["name"] for source in source_details],
        "source_details": source_details,
        "record_count": len(records),
    }
    if not records:
        result["warning"] = "Nenhum registro encontrado para este patient_id."
    return json.dumps(result, ensure_ascii=False, default=str)


def _sanitize_protocol_content(text: str, source_name: str) -> str:
    """Remove metadados, formatação markdown, e referências de arquivo do protocolo."""
    import re
    
    # Remove HTML comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    
    # Remove markdown headers e deixa só o texto (evita repetição de nomes de arquivo)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    
    # Remove markdown links mantendo só o texto
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Remove markdown bold/italic
    text = re.sub(r'[*_]{1,3}', '', text)
    
    # Remove linhas que mencionam extensões de arquivo
    lines = [line for line in text.split('\n') 
             if not re.search(r'\.md|\.csv|\.json|\.txt', line, re.IGNORECASE)]
    text = '\n'.join(lines)
    
    # Remove múltiplas quebras de linha
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove linhas vazias no início e fim
    text = text.strip()
    
    return text


def _protocol_documents() -> list[tuple[str, str]]:
    """Carrega protocolos médicos e sanitiza conteúdo para evitar vazamento de metadados."""
    if not PROTOCOLS_DIR.exists():
        return []
    documents = []
    for path in sorted(PROTOCOLS_DIR.rglob("*")):
        # O README descreve a pasta e não deve virar contexto clínico indexado.
        if path.name.lower() == "readme.md":
            continue
        # Apenas .txt é permitido para evitar exposição de formatação markdown
        if path.suffix.lower() == ".txt" and path.is_file():
            content = path.read_text(encoding="utf-8")
            content = _sanitize_protocol_content(content, path.name)
            if content:  # Só adiciona se houver conteúdo após sanitização
                documents.append((path.name, content))  # Use apenas o nome, não o path completo
        elif path.suffix.lower() == ".md" and path.is_file():
            # Se for .md, converte para texto puro removendo formatação
            content = path.read_text(encoding="utf-8")
            content = _sanitize_protocol_content(content, path.name)
            if content:  # Só adiciona se houver conteúdo após sanitização
                # Use apenas o nome base sem .md, e identifique como "Protocolo"
                clean_name = path.stem.replace("_", " ").title()
                documents.append((f"Protocolo: {clean_name}", content))
    return documents


def _protocol_metadata() -> dict[str, dict[str, Any]]:
    metadata_path = PROTOCOLS_DIR / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return {str(item.get("file")): item for item in metadata if item.get("file")}
    except (OSError, json.JSONDecodeError):
        return {}


def _rank_protocols(query: str, documents: list[tuple[str, str]]) -> list[dict[str, str]]:
    """Busca e classifica protocolos por relevância usando TF-IDF, sem expor paths."""
    chunks = []
    for source, text in documents:
        # Divide em parágrafos mantendo a origem clara mas sem paths do sistema
        paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
        for index, paragraph in enumerate(paragraphs, start=1):
            # Source agora é um nome amigável sem path do sistema
            chunks.append({"source": f"{source}", "content": paragraph, "order": index})
    if not chunks:
        return []

    vectorizer = TfidfVectorizer(lowercase=True)
    matrix = vectorizer.fit_transform([chunk["content"] for chunk in chunks])
    query_vector = vectorizer.transform([query])
    scores = (matrix @ query_vector.T).toarray().ravel()

    try:
        import faiss

        index = faiss.IndexFlatIP(matrix.shape[1])
        dense_matrix = matrix.toarray().astype("float32")
        index.add(dense_matrix)
        _, indexes = index.search(query_vector.toarray().astype("float32"), min(4, len(chunks)))
        selected = indexes[0]
    except ImportError:
        # O ranking por NumPy mantém a funcionalidade em instalações sem FAISS.
        selected = scores.argsort()[::-1][:4]

    return [
        {**chunks[index], "score": round(float(scores[index]), 4)}
        for index in selected
        if index >= 0 and scores[index] > 0
    ]


@tool
def search_medical_protocols(query: str) -> str:
    """Recupera trechos relevantes dos protocolos médicos locais sem expor metadados."""
    documents = _protocol_documents()
    if not documents:
        return json.dumps(
            {
                "results": [],
                "sources": [],
                "source_details": [],
                "warning": "Nenhum protocolo médico foi configurado em MEDICAL_PROTOCOLS_DIR.",
            },
            ensure_ascii=False,
        )

    results = _rank_protocols(query, documents)
    metadata = _protocol_metadata()
    source_details = []
    
    # Extrai apenas o nome do protocolo sem path completo
    seen_sources = set()
    for item in results:
        source_name = item["source"].split("(")[0].strip() if "(" in item["source"] else item["source"]
        if source_name not in seen_sources:
            source_details.append({
                "name": source_name,
                "type": "medical_protocol",
                "metadata": metadata.get(source_name, {}),
            })
            seen_sources.add(source_name)

    return json.dumps(
        {
            "results": results,
            "sources": [item["source"] for item in results],
            "source_details": source_details,
            "warning": None if results else "Nenhum trecho relevante foi encontrado nos protocolos.",
        },
        ensure_ascii=False,
    )


TOOLS = [search_patient_history, search_medical_protocols]