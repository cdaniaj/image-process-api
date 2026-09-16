# SDD 2.0 - Estado, Requisitos e Backlog da Image Process API

**Data:** 2026-09-06  
**Sistema:** `image-process-api`  
**Finalidade:** fornecer a outra LLM uma visão factual e verificável do sistema, dos requisitos do Tech Challenge Fase 3, do que já foi implementado e do que ainda precisa ser demonstrado ou corrigido.

> Este documento substitui a versão anterior do SDD de análise. O código real é a fonte de verdade; descrições antigas em outros documentos podem estar desatualizadas.

## 1. Objetivo do sistema

A Image Process API é um backend FastAPI para apoio à análise de imagens mamográficas/citológicas. O sistema combina:

1. processamento de imagem com OpenCV;
2. extração de características geométricas;
3. classificação com Random Forest e Logistic Regression;
4. geração de relatório textual por LLM;
5. confirmação humana e persistência em CSV;
6. consulta de histórico estruturado;
7. recuperação local de protocolos;
8. agente conversacional com LangChain/LangGraph;
9. pipeline preparado para fine-tuning QLoRA/PEFT;
10. logs e métricas Prometheus.

O sistema é um protótipo acadêmico de apoio à decisão. Não é um sistema de diagnóstico autônomo, não prescreve tratamentos e não deve ser apresentado como validado clinicamente.

## 2. Requisitos de referência

O SDD funcional do módulo médico exige:

- LLM ajustada ou pipeline equivalente integrado ao assistente;
- uso de protocolos médicos, FAQs e exemplos de laudos/procedimentos;
- preprocessing, anonimização e curadoria do dataset;
- consulta a dados estruturados de pacientes;
- contextualização com dados atualizados;
- orquestração LangChain;
- automação de fluxo LangGraph;
- guardrails contra prescrição e diagnóstico definitivo;
- logging detalhado para auditoria;
- explainability com fontes;
- dados anonimizados ou sintéticos;
- README com execução e configuração.

## 3. Status executivo

### 3.1 Implementado e validado

- API FastAPI e endpoint `/assistant/chat`.
- Tools LangChain para histórico e protocolos.
- Histórico sintético consultável por `patient_id`.
- RAG local lexical/TF-IDF com FAISS opcional.
- Workflow LangGraph linear com nós de validação, recuperação, geração e guardrails.
- Provider Gemini via `langchain-google-genai`.
- Provider local PEFT preparado via `ASSISTANT_MODEL_PROVIDER=fine_tuned`.
- Fallback offline e fallback quando Gemini falha/quota é excedida.
- Disclaimer obrigatório no contrato de resposta.
- `sources`, `source_details` e `context_used` na resposta.
- Validação de upload, tamanho e contorno nulo.
- Lock em memória para treinamento concorrente.
- Parser compatível com CSV confirmado legado e schemas mistos.
- Dataset sintético de histórico e exemplos conversacionais sintéticos.
- Docker build do backend e frontend.
- Suíte validada: `23 passed` em modo offline.

### 3.2 Implementado, mas não comprovado em ambiente final

- Treinamento real QLoRA/PEFT: depende de GPU/stack opcional e ainda requer execução com o dataset revisado.
- Inferência local com adapter PEFT: o código existe, mas ainda não há adapter treinado validado em `fine_tunning/model_output/adapter/`.
- RAG com protocolos oficiais: a estrutura está pronta, mas o diretório contém apenas README e `metadata.json` vazio.
- Uso clínico dos textos: não existe revisão médica formal dos dados sintéticos.

### 3.3 Ainda pendente

- Persistência de memória de chat além do processo.
- Autenticação/autorização.
- Banco transacional para histórico.
- Validação robusta de citações contra fontes recuperadas.
- Auditoria completa com correlation ID, duração e eventos por nó.
- Métricas de qualidade do fine-tuning.
- Política formal de retenção e proteção de dados.
- Workflow LangGraph condicional, com decisões de rota e alertas.

## 4. Arquitetura real atual

```text
Frontend Angular / Cliente HTTP / Swagger
                    |
                    v
                 main.py
                    |
     +--------------+----------------+
     |              |                |
 /analyze       /model       /assistant/chat
     |              |                |
 OpenCV       ML + AG      LangGraph workflow
     |              |                |
     |          artefatos       +-----+-------+
     |          joblib          |             |
     |                     histórico       protocolos
     |                     CSV/sintético  TF-IDF/FAISS
     |                            \       /
     |                             provider
     |                         Gemini ou PEFT
     |
 CSV, PNG, PKL, JSONL, logs, métricas Prometheus
```

### 4.1 Componentes e responsabilidades

| Componente | Arquivo/diretório | Estado |
|---|---|---|
| API | `main.py` | Rotas, CORS, upload e composição |
| Processamento | `normalization/`, `morphology/`, `contours/` | Implementado |
| Classificação | `ai_model/mamography_rf/` | RF, LR e AG implementados |
| Persistência | `data_extraction/extraction.py` | CSV confirmado e compatibilidade legada |
| Laudo | `llm_layer/client.py` | Gemini direto para o fluxo `/analyze` |
| Tools | `langchain_agent/tools.py` | Histórico e protocolos |
| Agente | `langchain_agent/agent.py` | Contrato público, memória e resposta |
| Workflow | `langgraph_workflows/assistant_graph.py` | LangGraph linear |
| Providers | `langchain_agent/providers.py` | Gemini, fallback e PEFT local preparado |
| Fine-tuning | `fine_tunning/train.py` | QLoRA/PEFT preparado |
| Dados sintéticos | `data/`, `fine_tunning/` | Demonstração, não revisados |
| Observabilidade | `observability/logger.py` | stdout e arquivo |
| Métricas | Instrumentator + Prometheus | Implementado |

## 5. Fluxos principais

### 5.1 Análise de imagem: `POST /analyze`

Entrada `multipart/form-data`:

- `file`;
- `patient_name`;
- `patient_id`.

Fluxo:

1. valida MIME `image/jpeg` ou `image/png`;
2. limita arquivo a 10 MB;
3. decodifica imagem em escala de cinza;
4. normaliza percentis 85/100;
5. aplica blur mediano;
6. aplica CLAHE;
7. binariza por Otsu;
8. aplica abertura/fechamento morfológico;
9. extrai contornos;
10. seleciona contorno mais próximo do centro;
11. calcula área, perímetro, circularidade, solidez, compactação, concavidade e raio;
12. executa RF/LR;
13. calcula consenso;
14. gera laudo via `llm_layer/client.py`;
15. salva imagem com contorno.

Erros esperados:

- `400`: imagem inválida;
- `413`: arquivo acima de 10 MB;
- `415`: MIME não permitido;
- `422`: nenhum contorno válido;
- `500`: falha de modelo, filesystem ou integração não tratada.

Limitações ainda relevantes:

- Gemini é chamado de forma síncrona dentro de rota `async`;
- o relatório inicial e o chat usam fluxos de LLM diferentes;
- o laudo inicial depende da chave Gemini no módulo legado;
- não há versionamento do modelo ML na resposta;
- o pipeline não é uma validação clínica.

### 5.2 Confirmação humana: `POST /confirm`

Recebe `PatientDiagnosticModel` e grava em `data_extraction/gerados/dataset_extraction.csv`.

A escrita atual inclui features, consenso, scores e previsões. O parser aceita arquivos antigos com nove colunas e linhas novas com quatorze colunas.

O campo `finalConsensus` possui os valores:

- `0`: ambos os modelos retornaram classe 0;
- `1`: ambos retornaram classe 1;
- `2`: modelos discordaram.

O valor `2` deve continuar sendo tratado como indefinido para treinamento clínico até existir regra explícita.

### 5.3 Treinamento ML: `POST /model`

O endpoint agenda `handleLearning` em background, retorna `202` e usa lock em memória.

O treinamento combina:

- `data.csv` legado;
- CSV confirmado, quando existente.

O histórico sintético em `data/synthetic_patient_history.csv` **não é incorporado automaticamente ao treinamento do Random Forest**.

O pipeline valida classes, converte features numéricas, executa baseline, três experimentos do algoritmo genético, RF final, LR, gráficos e serialização.

Limitações:

- lock não é distribuído entre múltiplos workers/containers;
- não há endpoint de status do treinamento;
- não há cancelamento;
- artefatos ainda não possuem escrita transacional completa;
- o teste de integração precisa mockar o treinamento para evitar execução pesada em cada suíte.

### 5.4 Assistente: `POST /assistant/chat`

Request:

```json
{
  "patient_id": "PATIENT_001",
  "query": "Resuma a evolução do paciente.",
  "session_id": "sessao-001"
}
```

Response atual:

```json
{
  "response": "Texto de apoio à decisão...",
  "sources": [
    "data/synthetic_patient_history.csv"
  ],
  "disclaimer": "Sugestão de IA para auxílio médico. Validação humana obrigatória.",
  "session_id": "sessao-001",
  "source_details": [
    {
      "name": "data/synthetic_patient_history.csv",
      "type": "synthetic_history",
      "synthetic": true
    }
  ],
  "context_used": {
    "patient_history": true,
    "medical_protocols": false,
    "synthetic_history": true
  }
}
```

Fluxo LangGraph atual:

```text
validate_input
  -> retrieve_history
  -> retrieve_protocols
  -> build_prompt
  -> generate_response
  -> validate_guardrails
  -> END
```

A memória é uma `deque` em processo, limitada a 12 mensagens por `session_id`. Ela é perdida ao reiniciar o processo/container.

## 6. Histórico e fontes

### 6.1 Fontes de histórico

A tool `search_patient_history` consulta:

1. `data.csv` legado, usando `id`;
2. `data_extraction/gerados/dataset_extraction.csv`, usando `patient_id`;
3. `data/synthetic_patient_history.csv`, usando `patient_id`.

A comparação atual é textual e exata. O histórico sintético é ordenado por `exam_date`.

### 6.2 Dados sintéticos

Arquivo:

```text
data/synthetic_patient_history.csv
```

Características:

- 30 registros;
- 10 pacientes fictícios;
- três exames por paciente;
- scores, labels, previsões e consensos;
- todos os registros com `synthetic=true`;
- todos com `review_status=not_reviewed`.

Esses dados servem para teste de integração e demonstração. Não são prontuários, evidência clínica ou diretriz.

### 6.3 Protocolos

Diretório:

```text
protocols/
```

Formato aceito pelo RAG atual:

- `.md`;
- `.txt`.

O `README.md` do diretório é ignorado. `metadata.json` existe, mas está vazio até que documentos reais sejam adicionados.

Metadados esperados por documento:

```json
{
  "file": "pcdt_cancer_mama.md",
  "title": "Título oficial",
  "institution": "Ministério da Saúde",
  "source_url": "https://...",
  "version": "2025",
  "published_at": "2025-01-01",
  "downloaded_at": "2026-09-06",
  "license": "Documento público",
  "review_status": "not_reviewed"
}
```

O sistema ainda não extrai PDF automaticamente. PDFs precisam ser convertidos para texto/Markdown mantendo o original para auditoria.

## 7. RAG atual

O RAG divide documentos em parágrafos, calcula TF-IDF e seleciona até quatro trechos. Se `faiss-cpu` estiver disponível, usa índice de produto interno; caso contrário, usa ordenação lexical pelo score TF-IDF.

Cada resultado pode incluir:

- trecho;
- arquivo/seção/parágrafo;
- score;
- metadados cadastrados em `protocols/metadata.json`.

Limitações:

- não há embeddings semânticos especializados;
- não há persistência do índice;
- o índice é reconstruído por consulta;
- não há OCR/PDF parser;
- ausência de metadata não bloqueia a fonte;
- não há validação de que a resposta textual cite apenas fontes recuperadas;
- não há reranking clínico ou filtro por versão vigente.

## 8. Fine-tuning

### 8.1 Pipeline existente

Arquivos:

- `fine_tunning/prepare_dataset.py`;
- `fine_tunning/train.py`;
- `fine_tunning/dataset_anonimizado.jsonl`;
- `fine_tunning/synthetic_history_conversations.jsonl`;
- `requirements-fine-tuning.txt`.

O pipeline:

1. lê JSONL;
2. formata mensagens;
3. tenta importar Datasets, PEFT, Transformers e Torch;
4. carrega `microsoft/phi-2`;
5. configura QLoRA 4-bit;
6. aplica LoRA em módulos de atenção;
7. tokeniza com labels causais;
8. executa `Trainer`;
9. salva adapter e tokenizer.

Se as dependências não existirem, retorna `demo_mode`.

### 8.2 Provider local

Provider configurado por:

```bash
ASSISTANT_MODEL_PROVIDER=gemini
```

ou:

```bash
ASSISTANT_MODEL_PROVIDER=fine_tuned
FINE_TUNING_OUTPUT_DIR=fine_tunning/model_output
```

O provider local valida:

- `adapter_config.json`;
- `adapter_model.safetensors` ou `adapter_model.bin`;
- `base_model_name_or_path` no config.

Dependências opcionais:

```text
torch
transformers
peft
datasets
accelerate
bitsandbytes
```

Estado atual: o código de carregamento está implementado, mas não há evidência registrada de um adapter treinado e executado com sucesso no ambiente final.

### 8.3 Dataset conversacional atual

O dataset original possui poucos exemplos e repete condutas semelhantes entre níveis de risco. O dataset derivado do histórico possui exemplos de:

- evolução;
- discordância;
- comparação de modelos;
- ausência de diagnóstico;
- recusa de prescrição;
- limitações do histórico.

Todos estão marcados como sintéticos e não revisados.

Requisitos para uma versão mais forte:

- 100-200 exemplos variados;
- fontes reais ou explicitamente sintéticas;
- revisão de conteúdo clínico;
- separação entre dados oficiais, sintéticos e públicos;
- versionamento do dataset;
- avaliação antes/depois do fine-tuning;
- conjunto de teste separado do conjunto de treino.

## 9. Providers e falhas

O provider Gemini captura exceções e retorna fallback. Isso evita 500 quando há quota excedida, timeout ou indisponibilidade.

O provider `fine_tuned` atualmente pode propagar erro se:

- dependências opcionais não estiverem instaladas;
- adapter não existir;
- modelo base não puder ser baixado;
- memória/GPU for insuficiente.

Isso precisa ser tratado antes de habilitar `fine_tuned` em produção, com resposta de erro explícita ou fallback configurável.

O modo de testes pode ser ativado por:

```bash
ASSISTANT_OFFLINE=true
```

## 10. Guardrails e segurança

### Implementado

- system prompt restritivo;
- disclaimer obrigatório;
- bloqueio por padrões proibidos;
- resposta segura em caso de bloqueio;
- declaração de ausência de histórico/protocolos;
- indicação de fontes e tipo sintético.

### Ainda insuficiente

O bloqueio é baseado em padrões simples e pode não detectar paráfrases. Faltam:

- classificador de segurança ou política estruturada;
- validação de presença do disclaimer na saída gerada;
- validação de fontes citadas contra `source_details`;
- detecção de diagnóstico implícito;
- detecção de recomendação terapêutica indireta;
- proteção específica contra prompt injection nos documentos recuperados;
- testes adversariais em português;
- auditoria de respostas bloqueadas.

## 11. Privacidade e LGPD

### Implementado parcialmente

- dados sintéticos separados;
- anonimização no preprocessing;
- hashes de pacientes no dataset de fine-tuning;
- indicação explícita de conteúdo sintético.

### Pendente

- não registrar `patient_id` completo em logs de produção;
- política de retenção;
- controle de acesso;
- criptografia em repouso/trânsito fora do TLS do ambiente;
- consentimento/base legal para dados reais;
- remoção de nomes do CSV operacional;
- segregação entre dado identificável e prompt enviado ao provider externo;
- documentação de fluxo de dados para Gemini;
- procedimento de exclusão/anonymização.

## 12. Observabilidade e auditoria

### Implementado

Logs registram eventos gerais da API, provider do assistente, fontes e bloqueio de guardrail. Prometheus expõe métricas HTTP.

### Pendente

Adicionar:

- `correlation_id` por requisição;
- `request_id` e `session_id` separados;
- duração total e por nó LangGraph;
- ferramenta executada e resultado resumido;
- número de registros recuperados;
- versão do protocolo/modelo;
- motivo do fallback;
- provider e modelo efetivos;
- status de guardrail;
- métricas específicas do assistente;
- logs sem dados pessoais diretos.

## 13. Frontend Angular

O frontend possui:

- upload e crop;
- `/analyze` real por padrão, com opção de mock;
- confirmação da análise;
- botão flutuante do assistente;
- histórico de mensagens em sessão;
- fontes e disclaimer básicos;
- chamada `POST /assistant/chat`.

Ainda falta exibir de forma completa:

- `source_details`;
- `context_used`;
- indicador de fonte sintética;
- provider usado;
- fallback/offline;
- estado de ausência de protocolos;
- aviso de resposta bloqueada por guardrail.

## 14. Testes atuais

A suíte validada contém 23 testes passando em modo offline.

Coberturas relevantes:

- histórico sintético;
- parser CSV legado;
- tools LangChain;
- workflow LangGraph;
- guardrail de prescrição;
- endpoint do assistente;
- modelos e predição;
- fine-tuning preprocessing;
- integração Gemini mockada;
- endpoints básicos.

Gaps de teste:

- provider fine-tuned com adapter real;
- erro de adapter ausente;
- falta de memória/GPU;
- quota Gemini 429;
- timeout do provider;
- prompt injection em protocolo;
- fonte citada que não existe;
- resposta sem disclaimer;
- múltiplos workers e lock distribuído;
- concorrência de escrita CSV;
- frontend exibindo `source_details`;
- PDFs e extração de metadata.

## 15. Matriz de conformidade com o SDD

| Requisito | Estado | Evidência/lacuna |
|---|---|---|
| API modular em Python | Atende | FastAPI e módulos existentes |
| Processamento de imagem | Atende | OpenCV + features |
| Modelos tradicionais | Atende | RF/LR/AG |
| Dados estruturados | Parcialmente atende | CSVs, sem banco transacional |
| Histórico por paciente | Atende para demonstração | `patient_id` e histórico sintético |
| RAG local | Parcialmente atende | TF-IDF/FAISS, sem protocolos oficiais |
| Tools LangChain | Atende | `@tool` em `tools.py` |
| Agente/orquestração LangChain | Parcialmente atende | provider e tools; decisão ainda linear |
| Workflow LangGraph | Atende estruturalmente | StateGraph linear implementado |
| Fine-tuning preprocessing | Parcialmente atende | anonimização/JSONL; dataset pequeno |
| QLoRA/PEFT | Preparado | não validado com adapter real |
| Fine-tuned integrado ao chat | Preparado, não comprovado | provider local implementado |
| Protocolos oficiais | Não atende ainda | `metadata.json` vazio |
| Guardrails de prompt | Atende parcialmente | system prompt |
| Guardrails técnicos | Atende parcialmente | padrões proibidos simples |
| Disclaimer | Atende | campo obrigatório |
| Explainability | Atende parcialmente | sources/source_details |
| Logging | Atende parcialmente | sem auditoria completa por nó |
| Privacidade | Atende parcialmente | dados sintéticos; sem governança completa |
| Frontend | Atende parcialmente | chat integrado, metadata ainda incompleta |
| Documentação | Atende parcialmente | README/SDD precisam refletir fontes reais |
| Testes | Atende parcialmente | 23 passam; faltam cenários de produção |

## 16. Backlog priorizado

### P0 - segurança e correção

1. Adicionar protocolos oficiais versionados e metadata real.
2. Impedir que o assistente trate dado sintético como evidência clínica.
3. Validar e bloquear citações inexistentes.
4. Tratar falha do provider fine-tuned com fallback/erro controlado.
5. Remover identificadores pessoais de logs.
6. Validar respostas sem disclaimer ou sem fonte.
7. Separar explicitamente dado de treinamento, histórico e protocolo.

### P1 - conformidade acadêmica

1. Executar treinamento PEFT real em ambiente compatível.
2. Registrar métricas, hardware, modelo base, dataset e versão do adapter.
3. Criar conjunto de avaliação separado.
4. Ampliar dataset conversacional.
5. Implementar testes adversariais de guardrails.
6. Criar decisões condicionais no LangGraph.
7. Atualizar frontend para `source_details/context_used`.
8. Adicionar endpoint/status de treinamento.

### P2 - produção

1. Migrar histórico para banco transacional.
2. Persistir memória por sessão.
3. Adicionar autenticação/autorização.
4. Implementar lock distribuído/fila.
5. Indexar PDFs com parser/OCR e embeddings persistentes.
6. Adicionar versionamento de modelos e protocolos.
7. Implementar política LGPD completa.
8. Adicionar tracing distribuído.

## 17. Plano de execução recomendado

### Etapa A - fontes oficiais

- adicionar PDFs/textos oficiais;
- extrair texto preservando documento original;
- preencher `protocols/metadata.json`;
- testar busca por termos conhecidos;
- verificar fontes retornadas.

### Etapa B - dados conversacionais

- revisar exemplos sintéticos;
- remover referências fictícias a protocolo interno;
- ampliar perguntas e respostas;
- separar train/validation/test;
- registrar versão e licença.

### Etapa C - fine-tuning

- instalar `requirements-fine-tuning.txt` em GPU/ambiente compatível;
- executar `train.py`;
- validar adapter;
- executar teste de geração local;
- comparar Gemini/fine-tuned;
- registrar métricas e limitações.

### Etapa D - segurança e auditoria

- validar fontes e disclaimer;
- testar prompt injection;
- testar prescrição/diagnóstico em múltiplas formulações;
- adicionar correlation ID;
- remover PII de logs.

### Etapa E - produto

- integrar metadata no frontend;
- documentar provider selecionado;
- adicionar status do treinamento;
- revisar Docker/volumes para adapter e protocolos;
- executar testes completos e smoke test do Compose.

## 18. Critérios de aceite finais

### Assistente

- `/assistant/chat` responde com `response`, `sources`, `source_details`, `context_used`, `session_id` e disclaimer.
- Uma pergunta sobre `PATIENT_001` retorna três registros sintéticos ordenados por data.
- A resposta deixa explícito quando a fonte é sintética.
- Ausência de histórico não gera falsa contextualização.
- Quota/timeout do Gemini não produz erro 500 quando fallback está habilitado.
- Provider fine-tuned ausente produz erro controlado ou fallback documentado.

### RAG

- Cada protocolo tem metadata de origem, instituição, versão e data.
- README não é indexado como diretriz.
- Trechos retornam caminho e seção/parágrafo.
- Fontes inexistentes não podem aparecer na resposta final.

### Fine-tuning

- Dataset possui preprocessing, anonimização e metadata.
- Treino gera adapter PEFT verificável.
- Adapter é carregado pelo provider local.
- Existe conjunto de avaliação separado.
- Há comparação documentada entre modelo base e fine-tuned.
- Falha sem GPU/dependências retorna `demo_mode` explicável.

### LangGraph

- Nós do workflow aparecem em testes.
- Falhas de input, retrieval, provider e guardrail têm comportamento definido.
- O grafo suporta ao menos uma decisão condicional baseada na intenção/contexto.
- Cada execução registra provider, fontes, duração e resultado do guardrail.

### Qualidade e privacidade

- Testes não dependem de API externa por padrão.
- Dados sintéticos não são misturados automaticamente ao classificador.
- Logs não expõem PII desnecessária.
- README e SDD refletem o estado real.

## 19. Instrução para outra LLM

Antes de editar:

1. leia este SDD;
2. confirme cada afirmação no código;
3. rode os testes direcionados;
4. identifique se o ambiente tem GPU, adapter e protocolos reais;
5. não trate `synthetic=true` como evidência clínica;
6. não invente fontes ou metadados;
7. preserve mudanças do usuário;
8. faça uma alteração pequena por vez;
9. execute validação focada após cada edição;
10. atualize este SDD somente com fatos verificáveis.

Ao responder uma análise, use este formato:

```text
Estado atual:
- atende / parcial / pendente

Evidência:
- arquivo, função, teste ou comando

Risco:
- impacto técnico, clínico ou de privacidade

Próxima ação:
- menor alteração testável

Dependência externa:
- informar somente se depender de fonte, credencial, GPU ou decisão do usuário
```

Não afirmar que o sistema possui fine-tuning validado apenas porque o provider existe. Não afirmar que possui protocolos oficiais enquanto `protocols/metadata.json` não estiver preenchido e os documentos não estiverem presentes.
