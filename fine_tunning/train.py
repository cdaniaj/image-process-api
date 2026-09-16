"""Treinamento opcional do adaptador PEFT usando LoRA otimizado para Apple Silicon (Mac MPS)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any
import torch


def load_dataset(dataset_path: str | Path) -> list[dict[str, Any]]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo {path} não encontrado.")

    rows: list[dict[str, Any]] = []
    # O formato JSONL permite processar um exemplo por linha e retomar a inspeção facilmente.
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


ASSISTANT_MARKER = "<|assistant|>\n"


def format_chat_example(example: dict[str, Any]) -> tuple[str, str]:
    """Formata o exemplo e retorna (texto_completo, prefixo_ate_o_assistente).

    O prefixo é usado para mascarar o loss em system/user: sem isso, o modelo gasta
    a maior parte do gradiente reaprendendo a pergunta em vez da resposta esperada.
    """
    messages = example.get("messages", [])
    blocks: list[str] = []
    prefix_blocks: list[str] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        block = f"<|{role}|>\n{content}\n"
        blocks.append(block)
        if role != "assistant":
            prefix_blocks.append(block)
    full_text = "\n".join(blocks)
    prefix_text = "\n".join(prefix_blocks) + "\n" + ASSISTANT_MARKER
    return full_text, prefix_text


def train_model(dataset_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Treina e salva o modelo de fine-tuning utilizando acelerador MPS no Mac ou CPU.

    Em ambientes sem o stack de treinamento completo, o método retorna um estado de demonstração,
    preservando o fluxo e evitando falhas na execução local.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:  # pragma: no cover - comportamento de ambiente limpo
        return {
            "status": "demo_mode",
            "message": (
                "Dependências de treinamento não instaladas; o dataset foi preparado para uso em ambiente "
                "com PyTorch/Transformers ou stack de fine-tuning completo."
            ),
            "error": str(exc),
        }

    dataset = load_dataset(dataset_path)
    pairs = [format_chat_example(record) for record in dataset]

    # Reserva os últimos exemplos como amostra de sanidade pós-treino (nunca entram no treino).
    holdout_size = min(3, max(1, len(pairs) // 20)) if len(pairs) >= 10 else 0
    train_pairs = pairs[: len(pairs) - holdout_size] if holdout_size else pairs
    holdout_pairs = pairs[len(pairs) - holdout_size :] if holdout_size else []

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Utilizando dispositivo para treinamento: {device}")

    model_name = "microsoft/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Carregamento otimizado para a memória unificada do Mac em float16 sem dependência de BitsAndBytes/CUDA
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(device)

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    max_length = int(os.getenv("FINE_TUNING_MAX_LENGTH", "512"))

    def build_features(record: dict[str, str]) -> dict[str, list[int]]:
        full = tokenizer(record["text"], truncation=True, max_length=max_length)
        prefix = tokenizer(record["prefix"], truncation=True, max_length=max_length)
        input_ids = full["input_ids"]
        # Mascara (-100) tudo antes da resposta do assistente: o loss só recai sobre o que o modelo deve gerar.
        prefix_len = min(len(prefix["input_ids"]), len(input_ids))
        labels = [-100] * prefix_len + input_ids[prefix_len:]
        return {"input_ids": input_ids, "attention_mask": full["attention_mask"], "labels": labels}

    dataset_hf = Dataset.from_list([{"text": text, "prefix": prefix} for text, prefix in train_pairs])
    tokenized = dataset_hf.map(build_features, remove_columns=["text", "prefix"])

    def data_collator(features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        pad_id = tokenizer.pad_token_id
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }

    training_args = TrainingArguments(
        output_dir=str(output_path),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        num_train_epochs=float(os.getenv("FINE_TUNING_EPOCHS", "3")),
        logging_steps=10,
        save_steps=50,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized, data_collator=data_collator)
    trainer.train()

    trainer.save_model(str(output_path / "adapter"))
    tokenizer.save_pretrained(str(output_path / "adapter"))

    sample_generations = _run_sanity_check(model, tokenizer, holdout_pairs, device)
    for sample in sample_generations:
        print(f"\n[sanidade] prompt: {sample['prompt'][:120]}...\n[sanidade] resposta: {sample['generated']}\n")

    return {
        "status": "success",
        "message": "Treinamento concluído com LoRA/PEFT via MPS no Mac.",
        "dataset_rows": len(dataset),
        "train_rows": len(train_pairs),
        "holdout_rows": len(holdout_pairs),
        "epochs": training_args.num_train_epochs,
        "output_dir": str(output_path),
        "sample_generations": sample_generations,
    }


def _run_sanity_check(model, tokenizer, holdout_pairs: list[tuple[str, str]], device: str) -> list[dict[str, str]]:
    """Gera respostas para exemplos nunca vistos no treino, para inspeção humana imediata da qualidade."""
    if not holdout_pairs:
        return []

    samples = []
    model.eval()
    with torch.no_grad():
        for _full_text, prefix_text in holdout_pairs:
            inputs = tokenizer(prefix_text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            output = model.generate(
                **inputs,
                max_new_tokens=200,
                min_new_tokens=20,
                do_sample=False,
                repetition_penalty=1.2,
                no_repeat_ngram_size=4,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            samples.append({"prompt": prefix_text, "generated": generated})
    model.train()
    return samples


def resolve_adapter_path(output_dir: str | Path | None = None) -> Path:
    """Resolve e valida a estrutura mínima de um adaptador PEFT salvo."""
    path = Path(output_dir or os.getenv("FINE_TUNING_OUTPUT_DIR", "fine_tunning/model_output")) / "adapter"
    if not path.is_dir():
        raise FileNotFoundError(f"Adaptador fine-tuned não encontrado: {path}")
    config = path / "adapter_config.json"
    weights = [path / "adapter_model.safetensors", path / "adapter_model.bin"]
    if not config.exists() or not any(weight.exists() for weight in weights):
        raise ValueError(f"Adaptador PEFT incompleto em {path}; esperados adapter_config.json e pesos.")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina um modelo de fine-tuning médico usando o dataset preparado.")
    parser.add_argument("--dataset", type=str, default=os.getenv("FINE_TUNING_DATASET", "fine_tunning/dataset_anonimizado.jsonl"), help="Dataset JSONL pronto para treinamento.")
    parser.add_argument("--output-dir", type=str, default=os.getenv("FINE_TUNING_OUTPUT_DIR", "fine_tunning/model_output"), help="Diretório para salvar o modelo ajustado.")
    args = parser.parse_args()

    result = train_model(args.dataset, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()