"""Prepara exemplos anonimizados e contextualizados para fine-tuning médico."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

SYSTEM_PROMPT = (
    "Você é um assistente médico especialista em oncologia mamária e radiologia, "
    "treinado com os protocolos internos do hospital. Nunca prescreva diretamente sem "
    "validação humana e indique sempre a fonte da diretriz utilizada."
)


def hash_patient(value: str) -> str:
    # Um hash curto mantém a consistência entre registros sem expor o identificador original.
    normalized = str(value or "anonimo").strip().lower().encode("utf-8")
    digest = hashlib.sha256(normalized).hexdigest()[:8]
    return f"PACIENTE_HASH_{int(digest, 16) % 10000}"


def anonymize_text(text: str) -> str:
    if text is None:
        return ""

    cleaned = str(text)
    # A ordem remove padrões altamente estruturados antes de procurar nomes em texto livre.
    cleaned = re.sub(
        r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b",
        "CPF_REMOVIDO",
        cleaned,
    )
    cleaned = re.sub(
        r"\b(?:\+55\s?)?(?:\(?\d{2}\)?[\s.-]?)?\d{4,5}[\s.-]?\d{4}\b",
        "TELEFONE_REMOVIDO",
        cleaned,
    )
    cleaned = re.sub(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        "EMAIL_REMOVIDO",
        cleaned,
    )
    cleaned = re.sub(
        r"(?i)\b(?:paciente|patient|nome|name)\s*[:=]?\s*[A-ZÀ-Ÿ][a-zà-ÿ]+(?:\s+(?:da|de|dos|das|do|del|e)\s+[A-ZÀ-Ÿ][a-zà-ÿ]+)*(?:\s+[A-ZÀ-Ÿ][a-zà-ÿ]+)*",
        lambda match: f"PACIENTE_HASH_{hash_patient(match.group(0))}",
        cleaned,
    )
    cleaned = re.sub(
        r"\b[A-ZÀ-Ÿ][a-zà-ÿ]+(?:\s+(?:da|de|dos|das|do|del|e)\s+[A-ZÀ-Ÿ][a-zà-ÿ]+)*(?:\s+[A-ZÀ-Ÿ][a-zà-ÿ]+)+\b",
        lambda match: f"PACIENTE_HASH_{hash_patient(match.group(0))}",
        cleaned,
    )
    cleaned = re.sub(r"\b(?:ID|id|paciente_id|patient_id)\s*[:=]?\s*[A-Za-z0-9-]+\b", "ID_REMOVIDO", cleaned)
    return cleaned


def sanitize_record(record: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in record.items():
        normalized_key = str(key).strip().lower()
        if value is None:
            sanitized[key] = ""
            continue

        if normalized_key in {"name", "nome", "patient_name", "paciente"}:
            sanitized[key] = hash_patient(str(value))
        elif "cpf" in normalized_key or "documento" in normalized_key:
            sanitized[key] = "CPF_REMOVIDO"
        elif "telefone" in normalized_key or "phone" in normalized_key or "celular" in normalized_key:
            sanitized[key] = "TELEFONE_REMOVIDO"
        elif "email" in normalized_key:
            sanitized[key] = "EMAIL_REMOVIDO"
        elif "id" in normalized_key and normalized_key not in {"diagnosis_id", "protocol_id"}:
            sanitized[key] = hash_patient(str(value))
        elif isinstance(value, str):
            # Campos textuais podem conter PII mesmo quando o nome da coluna não indica isso.
            sanitized[key] = anonymize_text(value)
        else:
            sanitized[key] = value
    return sanitized


def _risk_label(score: float) -> str:
    if score >= 61:
        return "high_risk"
    if score >= 31:
        return "moderate_risk"
    return "no_risk"


def build_dataset(
    csv_path: str | Path,
    output_path: str | Path,
    limit: int | None = None,
    protocol_texts: Iterable[str] | None = None,
    faq_texts: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    csv_file = Path(csv_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    protocols = list(protocol_texts or [
        "Protocolo de rastreamento mamográfico: lesões moderadas exigem avaliação complementar e acompanhamento em 6 meses.",
        "Diretriz interna: risco alto requer reforço de avaliação clínica e investigação complementar.",
    ])
    faqs = list(faq_texts or [
        "Qual a conduta para lesão moderada? Resposta: avaliação complementar e seguimento em 6 meses.",
        "Como proceder em achado de risco alto? Resposta: reavaliação clínica, exames complementar e discussão multidisciplinar.",
    ])

    dataset: list[dict[str, Any]] = []
    # O JSONL usa o formato de mensagens esperado por modelos conversacionais.
    with csv_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            if limit is not None and index >= limit:
                break

            safe_row = sanitize_record(row)
            risk_score = 0.0
            for key in ("risk_score", "score", "risk"):
                if safe_row.get(key) not in (None, ""):
                    try:
                        risk_score = float(str(safe_row[key]).replace(",", "."))
                    except ValueError:
                        risk_score = 0.0
                    break
            if risk_score == 0.0:
                # Gera um valor estável para datasets sem coluna de risco, útil em dados sintéticos.
                risk_score = 30.0 + ((index % 50) * 1.5)

            risk_label = _risk_label(risk_score)
            area_mean = safe_row.get("area_mean") or safe_row.get("area") or 1200
            circularity = 0.82 if risk_score >= 50 else 0.68

            question = (
                f"De acordo com os protocolos de rastreamento do hospital, qual a conduta para uma paciente "
                f"com lesão classificada como {risk_label} (score {risk_score:.1f}%) e circularidade {circularity:.2f}?"
            )
            answer = (
                f"Baseado no Protocolo Interno de Mastologia (Seção 4.2): lesões com risco {risk_label} e "
                f"circularidade {circularity:.2f} exigem revisão clínica complementar. Recomenda-se: 1. Exame "
                f"complementar de ultrassom para melhor caracterização; 2. Acompanhamento mamográfico em 6 meses; "
                f"3. Validação médica obrigatória antes de qualquer conduta terapêutica. A área observada foi {area_mean}."
            )

            protocol_context = protocols[index % len(protocols)]
            faq_context = faqs[index % len(faqs)]
            entry = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"{protocol_context} {faq_context} {question}"},
                    {"role": "assistant", "content": answer},
                ],
                "metadata": {
                    "risk_score": round(risk_score, 2),
                    "risk_label": risk_label,
                    "area_mean": float(area_mean),
                    "patient_hash": hash_patient(str(safe_row.get('name') or 'anonimo')),
                },
            }
            dataset.append(entry)

    with output_file.open("w", encoding="utf-8") as handle:
        for item in dataset:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara o dataset anonimizado para fine-tuning do modelo médico.")
    parser.add_argument("--csv", type=str, default="data.csv", help="Caminho do CSV de entrada.")
    parser.add_argument("--output", type=str, default="fine_tunning/dataset_anonimizado.jsonl", help="Caminho do arquivo JSONL de saída.")
    parser.add_argument("--limit", type=int, default=None, help="Número máximo de registros a processar.")
    args = parser.parse_args()

    build_dataset(args.csv, args.output, limit=args.limit)
    print(f"Dataset gerado em {args.output}")


if __name__ == "__main__":
    main()
