FROM python:3.11-slim

WORKDIR /app

# Dependencias de sistema requeridas por OpenCV y MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copiar código y modelos entrenados
COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
