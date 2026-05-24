import os
import time

os.environ.setdefault('GLOG_minloglevel', '2')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

import pickle
import socket
import threading
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from config import COUNTRIES, MODEL_BASE
from utils.hand_detector import extract_landmarks, mp_drawing, mp_hands

# ── Estado del modelo ────────────────────────────────────────────────────────
_model_lock = threading.Lock()
current_country = 'asl'
model = None
le = None


def load_model(country: str):
    global model, le, current_country
    classifier_path = MODEL_BASE / country / 'classifier.pkl'
    encoder_path    = MODEL_BASE / country / 'label_encoder.pkl'

    if not classifier_path.exists() or not encoder_path.exists():
        raise FileNotFoundError(
            f"No se encontró modelo para '{country}'. "
            f"Ejecuta: python model/train.py --country {country}"
        )

    with open(classifier_path, 'rb') as f:
        new_model = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        new_le = pickle.load(f)

    with _model_lock:
        model = new_model
        le = new_le
        current_country = country


def get_available_countries():
    available = {}
    for key, meta in COUNTRIES.items():
        model_dir = MODEL_BASE / key
        has_model = (model_dir / 'classifier.pkl').exists() and \
                    (model_dir / 'label_encoder.pkl').exists()
        available[key] = {**meta, 'available': has_model}
    return available


load_model('asl')

app = Flask(__name__)


# ── Utilidades de video ──────────────────────────────────────────────────────
def _placeholder_frame(message: str) -> bytes:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, message, (40, 240), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 0, 255), 2, cv2.LINE_AA)
    _, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()


# ── Cámara compartida (hilo de fondo) ───────────────────────────────────────
_cam_lock     = threading.Lock()
_cam_ready    = threading.Event()
_latest_annotated = None   # frame BGR con landmarks y predicción dibujados
_latest_landmarks = None   # np.array de 63 floats, o None si no hay mano
_camera_active    = False


def _start_camera():
    def worker():
        global _latest_annotated, _latest_landmarks, _camera_active
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not cam.isOpened():
            _camera_active = False
            _cam_ready.set()
            return

        # Windows necesita varios frames de calentamiento antes de devolver
        # datos válidos; leerlos sin procesar evita un break prematuro.
        for _ in range(10):
            cam.read()
            time.sleep(0.05)

        _camera_active = True
        _cam_ready.set()

        consecutive_failures = 0
        while True:
            ok, frame = cam.read()
            if not ok:
                consecutive_failures += 1
                if consecutive_failures > 60:   # ~2s de fallos consecutivos → desconectada
                    break
                time.sleep(0.033)
                continue
            consecutive_failures = 0

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lm, hand_lm = extract_landmarks(rgb)

            out = frame.copy()
            if lm is not None:
                with _model_lock:
                    m, enc = model, le
                label = enc.inverse_transform([m.predict([lm])[0]])[0]
                mp_drawing.draw_landmarks(out, hand_lm, mp_hands.HAND_CONNECTIONS)
                cv2.putText(out, label, (50, 150), cv2.FONT_HERSHEY_SIMPLEX,
                            3, (255, 0, 255), 4, cv2.LINE_AA)

            out = cv2.flip(out, 1)  # efecto espejo — más natural para el usuario

            with _cam_lock:
                _latest_annotated = out
                _latest_landmarks = lm

        cam.release()

    threading.Thread(target=worker, daemon=True).start()


# Solo abrir la cámara en el proceso que realmente sirve peticiones.
# En debug mode, Flask usa un reloader que crea DOS procesos; el proceso padre
# (watcher) no debe capturar la cámara para que el hijo (servidor) pueda hacerlo.
if os.environ.get('WERKZEUG_RUN_MAIN') != 'false':
    _start_camera()


# ── Generador MJPEG ──────────────────────────────────────────────────────────
def generate_frames():
    _cam_ready.wait(timeout=15.0)

    if not _camera_active:
        # Cámara no disponible: enviar placeholder en loop para mantener el stream abierto
        placeholder = _placeholder_frame('No se detecto camara')
        while True:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                   + placeholder + b'\r\n')
            time.sleep(1.0)
        return

    while True:
        with _cam_lock:
            frame = _latest_annotated
        if frame is None:
            time.sleep(0.033)
            continue
        ok, buf = cv2.imencode('.jpg', frame)
        if ok:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                   + buf.tobytes() + b'\r\n')
        time.sleep(0.033)


# ── Rutas principales ────────────────────────────────────────────────────────
@app.route('/')
def index():
    countries = get_available_countries()
    return render_template('index.html',
                           countries=countries,
                           current_country=current_country)


@app.route('/set_country', methods=['POST'])
def set_country():
    data = request.get_json()
    country = data.get('country', '').strip()

    if country not in COUNTRIES:
        return jsonify({'ok': False, 'error': 'País no válido'}), 400

    try:
        load_model(country)
        return jsonify({
            'ok': True,
            'country': country,
            'name': COUNTRIES[country]['name'],
            'flag': COUNTRIES[country]['flag'],
        })
    except FileNotFoundError as e:
        return jsonify({'ok': False, 'error': str(e)}), 404


@app.route('/current_country')
def get_current_country():
    return jsonify({'country': current_country, **COUNTRIES[current_country]})


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ── Rutas de aprendizaje ─────────────────────────────────────────────────────
@app.route('/learn')
def learn():
    countries = get_available_countries()
    with _model_lock:
        letters = sorted([c for c in le.classes_ if len(c) == 1 and c.isalpha()])
    return render_template('learn.html',
                           countries=countries,
                           current_country=current_country,
                           letters=letters)


@app.route('/learn/score')
def learn_score():
    letter = request.args.get('letter', '').upper()

    with _cam_lock:
        landmarks = _latest_landmarks

    if landmarks is None:
        return jsonify({'score': 0.0, 'detected': False, 'label': None})

    with _model_lock:
        m, enc = model, le

    if letter not in enc.classes_:
        return jsonify({'score': 0.0, 'detected': True, 'label': None})

    proba = m.predict_proba([landmarks])[0]
    target_idx = list(enc.classes_).index(letter)
    score = float(proba[target_idx])
    predicted_label = enc.classes_[proba.argmax()]

    return jsonify({'score': score, 'detected': True, 'label': predicted_label})


# ── Pista visual (landmarks de referencia desde los datos de entrenamiento) ──
@app.route('/learn/hint/<letter>')
def learn_hint(letter):
    import pandas as pd
    from config import DATA_SOURCES
    letter = letter.upper()

    with _model_lock:
        country = current_country

    dfs = []
    for path in DATA_SOURCES[country]:
        if path.exists():
            try:
                df = pd.read_csv(path)
                subset = df[df['label'] == letter]
                if not subset.empty:
                    dfs.append(subset)
            except Exception:
                pass

    if not dfs:
        return jsonify({'ok': False, 'error': f'Sin datos para {letter}'})

    df = pd.concat(dfs, ignore_index=True)
    points = [{'x': float(df[f'x{i}'].mean()), 'y': float(df[f'y{i}'].mean())}
              for i in range(21)]

    return jsonify({'ok': True, 'letter': letter, 'points': points})


# ── Dashboard de progreso ────────────────────────────────────────────────────
@app.route('/dashboard')
def dashboard():
    countries = get_available_countries()
    return render_template('dashboard.html',
                           countries=countries,
                           current_country=current_country)


# ── Modo deletrear palabras ──────────────────────────────────────────────────
SPELL_WORDS = {
    'asl':      ['LOVE', 'HELP', 'WATER', 'HAND', 'SIGN', 'LEARN',
                 'FOOD', 'HOME', 'GOOD', 'BOOK', 'PLAY', 'WORK'],
    'colombia': ['HOLA', 'AMOR', 'AGUA', 'MANO', 'LUNA', 'CASA',
                 'FLOR', 'BIEN', 'SOL', 'MAR', 'PAZ', 'LUZ'],
    'china':    ['LOVE', 'HELP', 'WATER', 'HAND', 'SIGN', 'LEARN',
                 'FOOD', 'HOME', 'GOOD', 'BOOK', 'PLAY', 'WORK'],
}

@app.route('/spell')
def spell():
    countries = get_available_countries()
    with _model_lock:
        country = current_country
        available = set(enc_class for enc_class in le.classes_)

    words = [w for w in SPELL_WORDS.get(country, SPELL_WORDS['asl'])
             if all(c in available for c in w)]

    return render_template('spell.html',
                           countries=countries,
                           current_country=country,
                           words=words)


# ── Arranque ─────────────────────────────────────────────────────────────────
def _lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(('8.8.8.8', 80))
            return s.getsockname()[0]
    except OSError:
        return '127.0.0.1'


if __name__ == '__main__':
    port = 5000
    print(f"\n  Local:   http://127.0.0.1:{port}")
    print(f"  Network: http://{_lan_ip()}:{port}\n")
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)
