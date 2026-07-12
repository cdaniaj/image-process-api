# Image Process API

API FastAPI para análise de imagens mamográficas com processamento de imagem, modelos de machine learning, geração de laudos com LLM e observabilidade.

## O que mudou recentemente

A API foi expandida para incluir:

- pipeline completo de análise de imagem com normalização, pré-processamento, binarização, morfologia e extração de contornos;
- endpoint de análise com upload de imagem e metadados do paciente;
- endpoint de confirmação para persistir amostras em CSV para treinamento futuro;
- endpoint de treinamento de modelos;
- integração com Gemini para geração de relatório médico textual;
- métricas Prometheus e logs estruturados para observabilidade;
- suporte a execução via Docker Compose com frontend, backend, Prometheus e Grafana;
- logs resilientes com fallback para diretórios graváveis caso o caminho padrão do container não esteja disponível.

## Estrutura atual do projeto

```text
image-process-api/
├── main.py
├── requirements.txt
├── Dockerfile
├── prometheus.yml
├── data.csv
├── ai_model/
│   └── mamography_rf/
│       ├── model.py
│       └── genetic_optimizer.py
├── contours/
│   └── contours.py
├── data_extraction/
│   └── extraction.py
├── dto/
│   └── patient_dto.py
├── llm_layer/
│   ├── client.py
│   └── prompt_templates.py
├── morphology/
│   └── morphology.py
├── normalization/
│   └── normalization.py
├── observability/
│   └── logger.py
├── outputfile/
│   └── outputfile.py
├── reports/
│   └── diagram.py
├── tests/
│   └── test_logger.py
└── logs/
```

## Como executar

### Opção 1: Docker Compose (recomendado)

Na raiz do repositório:

```bash
docker compose up -d --build
```

Serviços disponíveis:

- Backend: http://localhost:8000
- Swagger: http://localhost:8000/docs
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000
- Frontend: http://localhost:4200

Volumes mapeados:

- ./resultados_csv -> /app/data_extraction/gerados
- ./resultados_imagens -> /app/outputfile/gerados
- ./logs -> /app/logs

### Opção 2: Execução local

```bash
cd image-process-api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Endpoints principais

### POST /analyze
Analisa uma imagem mamográfica enviada pelo usuário.

Form-data esperado:

- file: imagem em JPG ou PNG
- patient_name: nome do paciente
- patient_id: identificador do paciente

Exemplo:

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@imagem.jpg" \
  -F "patient_name=Maria Silva" \
  -F "patient_id=12345"
```

Resposta inclui métricas geométricas, score de risco, rótulo de risco e o objeto completo do paciente com dados de predição.

### POST /confirm
Salva uma amostra confirmada no arquivo CSV de treinamento.

Exemplo:

```bash
curl -X POST http://localhost:8000/confirm \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Maria Silva",
    "id": "12345",
    "file_name": "imagem.jpg",
    "area_mean": 1234.56,
    "compactness_mean": 0.5,
    "perimeter_mean": 145.78,
    "concavity_mean": 0.3,
    "radius_mean": 25.5,
    "finalConsensus": 1,
    "risk_score": 0.0,
    "risk_label": "",
    "prediction": 0,
    "prediction_lr": 0,
    "risk_score_lr": 0,
    "llm_explanation": "",
    "llm_insights": ""
  }'
```

### POST /model
Treina os modelos de machine learning com os dados já confirmados.

```bash
curl -X POST http://localhost:8000/model
```

### POST /reports
Gera os artefatos visuais de relatório.

```bash
curl -X POST http://localhost:8000/reports
```

### GET /metrics
Expõe métricas para o Prometheus.

## Pipeline de processamento

O fluxo atual segue esta ordem:

1. leitura da imagem;
2. normalização de intensidade;
3. pré-processamento com blur mediano e CLAHE;
4. binarização via threshold de Otsu;
5. limpeza morfológica;
6. extração de contornos e cálculo de features;
7. predição com modelos de ML;
8. geração de laudo médico via LLM;
9. persistência de imagens e dados de saída.

## Dependências principais

- FastAPI
- Uvicorn
- OpenCV
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Pydantic
- python-multipart
- Prometheus FastAPI Instrumentator
- google-generativeai

## Variáveis de ambiente

Para a integração com o modelo de linguagem, configure a chave de acesso:

```bash
export GEMINI_API_KEY=sua_chave_aqui
```

Opcionalmente, você pode definir um diretório específico para os logs:

```bash
export IMAGE_PROCESS_LOG_DIR=/tmp/image-process-api/logs
```

## Observabilidade

A API registra logs de requisições e expõe métricas para monitoramento via Prometheus. O stack do Docker Compose já inclui:

- Prometheus em http://localhost:9090
- Grafana em http://localhost:3000

Credenciais padrão do Grafana:

- usuário: admin
- senha: admin123

Para acompanhar os logs do backend em tempo real:

```bash
docker compose logs -f backend
```

Ou, localmente:

```bash
tail -f logs/app_performance.log
```

## Observações

- As imagens processadas e os artefatos gerados são salvos em pastas sob os diretórios de saída do projeto.
- Para ajustar CORS ou outras políticas da API, consulte o arquivo main.py.
