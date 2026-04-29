# 1. Imagem base leve com Python
FROM python:3.10-slim

# 2. Instalação de dependências de sistema para o OpenCV rodar no Linux
# Isso é essencial para evitar erros de bibliotecas compartilhadas (.so)
# Instalamos as dependências modernas para OpenCV no Debian Trixie/Bookworm
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Define o diretório de trabalho dentro do container
WORKDIR /app

# 4. Copia os requisitos e instala as bibliotecas Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copia todo o código do seu backend para dentro do container
# Incluindo seus scripts de extração de ROI e pré-processamento
COPY . .

# 6. Porta que o FastAPI utilizará (padrão 8000)
EXPOSE 8000

# 7. Comando para iniciar o servidor
# O --host 0.0.0.0 é obrigatório para o container aceitar conexões externas
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]