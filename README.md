# sign-language-detector

Detector de lenguaje de señas en tiempo real con Flask + MediaPipe + scikit-learn. Incluye plataforma educativa, dashboard de progreso y modo deletrear.

## Requisitos

- **Python 3.11** (obligatorio — los wheels de MediaPipe 0.10.14 están compilados para `cp311`. En 3.13+ fallará la instalación).

## Instalación local

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

En Linux/macOS:
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Ejecución local

```powershell
python app.py
```

| URL | Página |
|---|---|
| `http://127.0.0.1:5000` | Detector en tiempo real |
| `http://127.0.0.1:5000/learn` | Flash cards con estrellas y pistas |
| `http://127.0.0.1:5000/dashboard` | Progreso y estadísticas |
| `http://127.0.0.1:5000/spell` | Modo deletrear palabras |

## Docker

La imagen está basada en `python:3.11-slim` y usa `opencv-contrib-python-headless` (sin dependencias de display).

### Construir la imagen

```bash
docker build -t sign-detector .
```

### Correr con Docker Compose (recomendado en Linux)

```bash
docker compose up
```

Abre `http://localhost:5000` en el navegador.

### Correr con Docker directamente

```bash
docker run -p 5000:5000 sign-detector
```

### Acceso a la cámara en Docker

El acceso a la cámara varía según el sistema operativo del host:

| Host | Cámara en Docker | Solución |
|---|---|---|
| **Linux** | Funciona con passthrough | `docker compose up` ya incluye `devices: /dev/video0` |
| **Windows** | No disponible | Docker corre sobre WSL2 sin acceso a cámaras USB del host |
| **macOS** | No disponible | Docker Desktop no expone cámaras al contenedor |

**En Windows y macOS**, la app arrancará pero mostrará el mensaje "No se detectó cámara" en el stream de video. El resto de la interfaz (dashboard, modo deletrear, historial) funciona normalmente.

Para usar la cámara en Windows/macOS, ejecuta la app directamente con Python (fuera de Docker):

```powershell
python app.py
```

## Entrenar el modelo

```powershell
# Capturar datos (~100 muestras por letra)
python capture_data.py --country asl

# Entrenar
python model/train.py --country asl
```

Los modelos se guardan en `model/<country>/classifier.pkl` y `model/<country>/label_encoder.pkl`.

## Notas importantes

- **No instales `mediapipe` sin pin de versión.** La versión 0.10.35+ eliminó la API `mp.solutions` en Windows. Usa siempre `pip install -r requirements.txt`.
- Si aparece `AttributeError: module 'mediapipe' has no attribute 'solutions'`:
  ```powershell
  pip install --force-reinstall --no-cache-dir mediapipe==0.10.14
  ```
