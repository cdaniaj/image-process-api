# 📊 Image Process API

API FastAPI para processamento e análise de imagens mamográficas com detecção de lesões utilizando visão computacional e machine learning.

## 📁 Estrutura do Projeto

```
image-process-api/
├── main.py                      # Arquivo principal da aplicação FastAPI
├── requirements.txt             # Dependências do projeto
├── Dockerfile                   # Configuração para containerização
├── data.csv                     # Dataset com dados dos pacientes
├── logistic_model.pkl           # Modelo treinado - Regressão Logística
├── trained_model.pkl            # Modelo treinado - Random Forest
├── scaler.pkl                   # Scaler para normalização dos dados
│
├── ai_model/                    # 🤖 Módulo de Modelos de IA
│   └── mamography_rf/
│       └── model.py             # Handlers para treinamento e predição de modelos
│
├── data_extraction/             # 📊 Extração de Características
│   └── extraction.py            # Lógica para preencher CSV com dados extraídos
│
├── dto/                         # 🔄 Data Transfer Objects
│   └── __init__.py              # Modelos Pydantic para transferência de dados
│
├── contours/                    # 🎯 Detecção e Análise de Contornos
│   └── __init__.py              # Funções para extrair contornos e calcular features
│
├── morphology/                  # 🔧 Operações Morfológicas
│   └── __init__.py              # Dilatação, erosão e limpeza de imagens
│
├── normalization/               # ⚙️ Normalização de Imagens
│   └── __init__.py              # Padronização de pixel values
│
├── outputfile/                  # 📁 Saída de Arquivos
│   └── __init__.py              # Geração de imagens processadas
│
└── reports/                     # 📈 Relatórios
    └── diagram.py               # Geração de gráficos e diagramas

```

## 🔍 Descrição de Cada Módulo

### `main.py` - Aplicação Principal
Arquivo central que configura a API FastAPI com os seguintes endpoints:

- **POST /analyze** - Analisa uma imagem mamográfica
  - Recebe: arquivo de imagem, nome do paciente, ID do paciente
  - Processa a imagem através de pipeline de visão computacional
  - Retorna: características extraídas, risco e predições

- **POST /confirm** - Confirma e salva dados do paciente
  - Preenche o arquivo data.csv com informações validadas

- **POST /model** - Treina o modelo de IA
  - Executa o treinamento dos modelos (Random Forest e Regressão Logística)

- **POST /reports** - Gera relatórios visuais
  - Cria gráficos e diagramas dos resultados

### `ai_model/mamography_rf/model.py` - Machine Learning
- **handleLearning()**: Treina os modelos de ML com dados do CSV
- **handlePrediction()**: Realiza predições usando os modelos treinados
- Utiliza Random Forest e Regressão Logística
- Calcula scores de risco e probabilidades

### `data_extraction/extraction.py` - Extração de Dados
- **fill_out_csv()**: Salva características extraídas no CSV
- Registra: nome, ID, características da lesão, nome do arquivo
- Prepara dados para treinamento do modelo

### `dto/__init__.py` - Modelos de Dados
Define a classe `PatientDiagnosticModel` com Pydantic:
- Dados do paciente (nome, ID)
- Características extraídas (área, compacidade, perímetro, etc.)
- Predições e scores de risco
- **convert_to_dto()**: Converte dados para o objeto DTO
- **get_risk_label()**: Classifica o nível de risco

### `contours/__init__.py` - Detecção de Contornos
- **get_contours()**: Extrai contornos da imagem binarizada
- **get_contours_data()**: Calcula features para cada contorno
- Features extraídas:
  - Área (area)
  - Compacidade (compactness)
  - Perímetro (perimeter)
  - Concavidade (concavity)
  - Raio médio (radius_mean)

### `morphology/__init__.py` - Operações Morfológicas
- **apply_morphology()**: Aplica operações de limpeza
- Realiza: erosão, dilatação e abertura morfológica
- Remove ruído e conecta estruturas fragmentadas

### `normalization/__init__.py` - Normalização
- **normalize_image()**: Normaliza valores de pixel
- Ajusta o contraste preservando estruturas importantes
- Parâmetros: mínimo e máximo percentil

### `outputfile/__init__.py` - Saída Processada
- **get_file()**: Salva imagens processadas
- Gera arquivos para visualização e debug
- Inclui: imagem processada com contornos sobrepostos

### `reports/diagram.py` - Relatórios
- **getReports()**: Cria visualizações dos resultados
- Gera gráficos de análise e estatísticas

## 🚀 Como Executar

### Pré-requisitos
- Docker instalado ([Download Docker](https://www.docker.com/products/docker-desktop))
- Docker Compose (opcional, vem com Docker Desktop)

### Opção 1: Executar com Docker (Recomendado)

#### 1️⃣ Build da Imagem
Construa a imagem Docker:
```bash
docker build -t image-process-api:latest .
```

#### 2️⃣ Executar o Container
Execute o container com a imagem criada:
```bash
docker run -p 8000:8000 --name image-api image-process-api:latest
```

**Explicação dos parâmetros:**
- `-p 8000:8000` - Mapeia a porta 8000 do container para a porta 8000 do seu computador
- `--name image-api` - Dá um nome ao container (opcional)
- `-d` - Executa em background (opcional, adicione para soltar o terminal)

#### 3️⃣ Acessar a API
Abra seu navegador ou cliente HTTP:
- **URL Base**: `http://localhost:8000`
- **Documentação Interativa**: `http://localhost:8000/docs` (Swagger UI)
- **Documentação Alternativa**: `http://localhost:8000/redoc` (ReDoc)

#### 4️⃣ Parar o Container
```bash
docker stop image-api
```

#### 5️⃣ Remover o Container
```bash
docker rm image-api
```

---

### Opção 2: Executar Localmente (sem Docker)

#### 1️⃣ Instalar Dependências
```bash
pip install -r requirements.txt
```

#### 2️⃣ Executar a Aplicação
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## 📋 Pipeline de Processamento de Imagem

O processo de análise segue estas etapas:

```
1. UPLOAD DA IMAGEM
   ↓
2. NORMALIZAÇÃO
   └─ Ajusta níveis de pixel (percentis 85-100)
   ↓
3. PRÉ-PROCESSAMENTO
   ├─ Blur Mediano (5x5)
   ├─ CLAHE (Contrast Limited Adaptive Histogram Equalization)
   ↓
4. BINARIZAÇÃO
   └─ Threshold de Otsu (OTSU automático)
   ↓
5. LIMPEZA MORFOLÓGICA
   ├─ Erosão
   ├─ Dilatação
   ├─ Abertura morfológica
   ↓
6. EXTRAÇÃO DE CONTORNOS
   ├─ Detecção de contornos
   ├─ Cálculo de features (área, perímetro, etc.)
   ↓
7. PREDIÇÃO DE IA
   ├─ Random Forest Classifier
   ├─ Logistic Regression
   ├─ Cálculo de risk scores
   ↓
8. RESPOSTA
   └─ Retorna análise completa ao usuário
```

---

## 🔗 Endpoints Principais

### POST /analyze
Analisa uma imagem mamográfica enviada.

**Request:**
```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@imagem.jpg" \
  -F "patient_name=João Silva" \
  -F "patient_id=12345"
```

**Response:**
```json
{
  "name": "João Silva",
  "id": "12345",
  "area": 1234.56,
  "perimeter": 145.78,
  "circularity": 0.89,
  "solidity": 0.92,
  "risk_score": 0.45,
  "risk_label": "Baixo Risco",
  "patient_data": {...}
}
```

### POST /confirm
Confirma dados do paciente e salva no CSV.

**Request:**
```bash
curl -X POST http://localhost:8000/confirm \
  -H "Content-Type: application/json" \
  -d '{
    "name": "João Silva",
    "id": "12345",
    "file_name": "imagem.jpg",
    "area_mean": 1234.56,
    "compactness_mean": 0.5,
    "perimeter_mean": 145.78,
    "concavity_mean": 0.3,
    "radius_mean": 25.5
  }'
```

### POST /model
Treina os modelos de machine learning.

```bash
curl -X POST http://localhost:8000/model
```

### POST /reports
Gera relatórios visuais.

```bash
curl -X POST http://localhost:8000/reports
```

---

## 📚 Dependências Principais

| Pacote | Versão | Função |
|--------|--------|--------|
| **fastapi** | 0.128.8 | Framework web |
| **uvicorn** | 0.39.0 | Servidor ASGI |
| **opencv-python** | 4.13.0 | Processamento de imagens |
| **numpy** | 2.0.2 | Computação numérica |
| **pandas** | 2.3.3 | Manipulação de dados |
| **scikit-learn** | 1.5.0 | Machine Learning |
| **matplotlib** | 3.9.4 | Visualização |
| **pydantic** | 2.12.5 | Validação de dados |

---

## 🐳 Explicação do Dockerfile

```dockerfile
# 1. Usa imagem Python 3.10 slim (mínima)
FROM python:3.10-slim

# 2. Instala dependências de sistema necessárias para OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \                    # Renderização gráfica
    libglib2.0-0 \              # Biblioteca GLIB
    && rm -rf /var/lib/apt/lists/*

# 3. Define diretório de trabalho no container
WORKDIR /app

# 4. Copia e instala dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copia todo o código da aplicação
COPY . .

# 6. Expõe a porta 8000
EXPOSE 8000

# 7. Inicia o servidor FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## ⚙️ Variáveis de Ambiente

Se necessário configurar CORS ou outras settings, edite o arquivo `main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Altere conforme necessário
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 🐛 Troubleshooting

### Erro: "docker: command not found"
- Instale Docker Desktop em https://www.docker.com

### Erro: "Port 8000 already in use"
- Use outra porta: `docker run -p 8001:8000 image-process-api:latest`
- Ou pare containers em execução: `docker ps` e `docker stop <container_id>`

### Erro: "ModuleNotFoundError"
- Certifique-se de que todos os arquivos estão no diretório raiz
- Verifique se `requirements.txt` está atualizado

---

## 📝 Notas Importantes

- Sempre valide as imagens antes do envio (formato, tamanho)

---

## 📧 Suporte

Para dúvidas ou problemas, abra uma issue no repositório.
