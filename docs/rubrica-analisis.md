# Análisis Rúbrica vs. Código — Sign Language Detector

> Última actualización: 2026-05-27

---

## Estado general

| Criterio | Peso | Estado |
|---|---|---|
| 1. Alcance y Dataset | 10% | PARCIAL |
| 2. Modelo de IA y Transfer Learning | 20% | PARCIAL / BRECHA CRÍTICA |
| 3. Backend y API | 20% | COMPLETO |
| 4. Interfaz y Visualización | 20% | COMPLETO |
| 5. Persistencia de Datos | 15% | FALTANTE / BRECHA CRÍTICA |
| 6. Dockerización y Orquestación | 15% | PARCIAL |

---

## Criterio 1 — Alcance y Manejo del Dataset (10%)

### Selección y Coherencia — COMPLETO
- Dataset ASL público (`data/archive/asl_landmarks_final.csv`) + capturas propias para Colombia y China via `capture_data.py`.
- Perfectamente alineado con la problemática de reconocimiento de señas.

### Formateo de Entrada — PARCIAL
- `utils/hand_detector.py:21` extrae 63 floats crudos (21 landmarks × x, y, z).
- MediaPipe los normaliza internamente a [0,1] en x,y, pero **no hay normalización adicional** antes del clasificador.
- La rúbrica pide "normalización de tensores según restricciones del modelo pre-entrenado" — aplica especialmente si se migra a CNN.
- **Pendiente:** agregar normalización/estandarización (StandardScaler o similar) antes del clasificador.

---

## Criterio 2 — Modelo de IA y Transfer Learning (20%) ⚠️ BRECHA CRÍTICA

### Estrategia de Fine-Tuning — FALTANTE
- El proyecto usa `RandomForestClassifier` de scikit-learn (`model/train.py:56`).
- **No es transfer learning.** MediaPipe es un modelo pre-entrenado (detección de landmarks), pero el clasificador final no carga pesos pre-entrenados ni hace fine-tuning.
- La rúbrica pide explícitamente "carga de pesos pre-entrenados mediante librerías abiertas" (TensorFlow, PyTorch, HuggingFace).
- **Pendiente:** reemplazar o complementar el RandomForest con un clasificador basado en red neuronal (MLP con Keras o MobileNetV2 fine-tuned).

### Métricas de Evaluación — PARCIAL
- `model/train.py:60-61` imprime accuracy y classification report **solo en consola** al entrenar.
- No hay: matriz de confusión guardada como imagen, curvas de pérdida/precisión, ni visualización en la UI.
- RandomForest no genera curvas de pérdida por epoch (otra razón para migrar a red neuronal).
- **Pendiente:** generar matriz de confusión como PNG (`model/<country>/confusion_matrix.png`) y exponerla desde el backend; mostrarla en el dashboard.

---

## Criterio 3 — Arquitectura del Backend y API (20%) ✅ COMPLETO

### Lógica de Servidor en Python — COMPLETO
- Flask 3.1, perfectamente modular: `config.py` → `utils/hand_detector.py` → `model/train.py` → `app.py`.
- Thread-safe con `_model_lock`.

### Endpoints de Procesamiento — COMPLETO
- `POST /predict` — recibe frame binario (base64 JSON) → inferencia → devuelve `{detected, label, score, landmarks}`.
- `POST /set_country` — hot-swap de modelo.
- `GET /current_country` — metadata JSON.
- `GET /learn/hint/<letter>` — coordenadas de referencia JSON.
- `GET /learn`, `/dashboard`, `/spell` — renders HTML.

---

## Criterio 4 — Interfaz de Usuario, Visualización y Captura (20%) ✅ COMPLETO

### Captura desde Dispositivo — COMPLETO
- `getUserMedia` en browser (`templates/index.html:168`), frame capturado en canvas, enviado como JPEG base64 cada 200ms a `/predict`.

### Experiencia de Visualización — COMPLETO
- Tema dark/neon, 4 páginas (detector, aprender, dashboard, deletrear), overlays de carga, toasts, flash cards con sistema de estrellas.

### Presentación de Resultados — COMPLETO
- Letra detectada superpuesta sobre video (`templates/index.html:219`).
- Landmarks con conexiones dibujadas en canvas overlay.
- Score bar en tiempo real en `/learn`.
- Gráfica A-Z con Chart.js en dashboard.

---

## Criterio 5 — Persistencia de Datos (15%) ❌ FALTANTE

### Almacenamiento del Historial — FALTANTE
- **Todo el historial vive en `localStorage` del browser** (clave `signdetect_history`).
- No hay ninguna conexión a base de datos en `app.py`.
- Si el usuario borra caché, pierde todo su historial.
- La rúbrica pide "conexión e interacción fluida desde el backend hacia motores de bases de datos (como MongoDB u otros)".
- **Pendiente:** agregar `pymongo`, colección `sessions` en MongoDB.

### Trazabilidad Técnica — FALTANTE
- El endpoint `/predict` procesa y descarta — no guarda nada.
- No hay registro de inferencias con etiqueta, score de confianza y timestamp en almacenamiento del servidor.
- **Pendiente:** colección `inferences` (campos: `letter`, `score`, `country`, `timestamp`), insertar en cada llamada a `/predict`. Agregar endpoints `GET /api/history` y `DELETE /api/history` que sirvan datos reales desde MongoDB al dashboard.

---

## Criterio 6 — Dockerización y Orquestación (15%) ⚠️ PARCIAL

### Aislamiento en Dockerfiles — COMPLETO
- `Dockerfile` con `python:3.11-slim`, instala `libgl1`/`libglib2.0-0`/`libgomp1`.
- Usa `requirements-docker.txt` separado del `requirements.txt` local.

### Orquestación Multicapa — PARCIAL
- `docker-compose.yml` existe pero tiene **solo 1 servicio** (`sign-detector`).
- Falta: servicio `mongodb`, red interna entre los dos contenedores, archivo `.env` con credenciales (`MONGO_URI`, `MONGO_USER`, `MONGO_PASS`), y que el compose referencie ese `.env`.
- **Pendiente:** agregar servicio `mongodb` al compose, crear `.env` con credenciales, usar `env_file: .env` en el compose.

---

## Brechas por prioridad de implementación

| # | Qué falta | Archivos afectados | Peso |
|---|---|---|---|
| 1 | **MongoDB** — colección `inferences`, endpoints `/api/history`, guardar cada predicción con label + score + timestamp | `app.py`, `docker-compose.yml`, `.env` | 15% |
| 2 | **Transfer Learning** — usar modelo pre-entrenado (MobileNetV2/MLP con Keras) en lugar de o junto al RandomForest | `model/train.py`, `requirements.txt` | 20% |
| 3 | **docker-compose multicapa** — agregar servicio `mongodb` + archivo `.env` + red interna | `docker-compose.yml`, `.env` | parte 15% |
| 4 | **Métricas visuales** — guardar matriz de confusión como PNG y exponerla en el dashboard | `model/train.py`, `app.py`, `templates/dashboard.html` | parte 20% |
| 5 | **Normalización de features** — centrar/escalar los 63 landmarks antes del clasificador | `model/train.py`, `utils/hand_detector.py` | parte 10% |

---

## Registro de cambios

| Fecha | Cambio |
|---|---|
| 2026-05-27 | Análisis inicial generado |
