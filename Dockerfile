FROM python:3.11-slim

WORKDIR /app

# Dependencias de sistema requeridas por OpenCV y MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copiar código y modelos entrenados
COPY . .

EXPOSE 10000

CMD gunicorn app:app --workers 1 --timeout 120 --bind 0.0.0.0:${PORT:-10000}
