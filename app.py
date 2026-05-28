import os
import sys
import json
import base64

os.environ.setdefault('GLOG_minloglevel', '2')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import csv as csv_module
import pickle
import threading
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

import cv2
import numpy as np
from flask import (Flask, jsonify, redirect, render_template,
                   request, session as flask_session, url_for)

from config import COUNTRIES, MODEL_BASE
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

HINT_IMG_DIRS = {
    'asl':      BASE_DIR / 'img' / 'señas ingles',
    'colombia': BASE_DIR / 'img' / 'señas español',
    'china':    BASE_DIR / 'img' / 'señas chino',
}
from utils.hand_detector import extract_landmarks
from utils.db import (authenticate_user, create_user,
                      delete_history, get_history, save_session)

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


TRAINING_ALLOWED_EMAIL = 'dammarts01@gmail.com'

load_model('asl')

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-change-me')
app.jinja_env.globals['TRAINING_ALLOWED_EMAIL'] = TRAINING_ALLOWED_EMAIL


# ── Auth helper ──────────────────────────────────────────────────────────────
def get_current_user():
    uid = flask_session.get('user_id')
    if not uid:
        return None
    return {
        'id':       uid,
        'username': flask_session.get('username'),
        'email':    flask_session.get('email', ''),
    }


# ── Rutas de autenticación ───────────────────────────────────────────────────
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html', error=None)

    username = request.form.get('username', '').strip()
    email    = request.form.get('email', '').strip()
    password = request.form.get('password', '')
    confirm  = request.form.get('confirm_password', '')

    if not username or not email or not password:
        return render_template('register.html', error='Todos los campos son obligatorios')
    if password != confirm:
        return render_template('register.html', error='Las contraseñas no coinciden')
    if len(password) < 6:
        return render_template('register.html', error='La contraseña debe tener al menos 6 caracteres')

    user_id, err = create_user(username, email, password)
    if err:
        return render_template('register.html', error=err)

    return redirect(url_for('login', registered='1'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        registered = request.args.get('registered')
        return render_template('login.html', error=None, registered=registered)

    identifier = request.form.get('identifier', '').strip()
    password   = request.form.get('password', '')

    if not identifier or not password:
        return render_template('login.html', error='Completa todos los campos', registered=None)

    user, err = authenticate_user(identifier, password)
    if err:
        return render_template('login.html', error=err, registered=None)

    flask_session['user_id']  = user['id']
    flask_session['username'] = user['username']
    flask_session['email']    = user.get('email', '')
    return redirect(url_for('index'))


@app.route('/logout', methods=['POST'])
def logout():
    flask_session.clear()
    return redirect(url_for('index'))


@app.route('/me')
def me():
    return jsonify({'user': get_current_user()})


# ── Predicción desde frame enviado por el browser ───────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or {}
    frame_b64 = data.get('frame', '')
    letter = data.get('letter', '').upper()

    try:
        frame_bytes = base64.b64decode(frame_b64)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return jsonify({'detected': False})

    if frame is None:
        return jsonify({'detected': False})

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    lm, hand_lm = extract_landmarks(rgb)

    if lm is None:
        return jsonify({'detected': False, 'label': None, 'score': 0.0, 'landmarks': None})

    with _model_lock:
        m, enc = model, le

    proba = m.predict_proba([lm])[0]
    label = enc.classes_[proba.argmax()]

    landmarks = [{'x': float(pt.x), 'y': float(pt.y)} for pt in hand_lm.landmark]

    score_val = float(proba.max())
    if letter and letter in enc.classes_:
        target_idx = list(enc.classes_).index(letter)
        score_val = float(proba[target_idx])

    return jsonify({
        'detected': True,
        'label': label,
        'score': score_val,
        'landmarks': landmarks,
    })


# ── Rutas principales ────────────────────────────────────────────────────────
@app.route('/')
def index():
    countries = get_available_countries()
    return render_template('index.html',
                           countries=countries,
                           current_country=current_country,
                           current_user=get_current_user())


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


# ── Rutas de aprendizaje ─────────────────────────────────────────────────────
@app.route('/learn')
def learn():
    countries = get_available_countries()
    with _model_lock:
        letters = sorted([c for c in le.classes_ if len(c) == 1 and c.isalpha()])
    return render_template('learn.html',
                           countries=countries,
                           current_country=current_country,
                           letters=letters,
                           current_user=get_current_user())


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


@app.route('/hint_img/<country>/<letter>')
def hint_img(country, letter):
    if country not in HINT_IMG_DIRS:
        return '', 404
    img_path = HINT_IMG_DIRS[country] / f'{letter.upper()}.jpg'
    if not img_path.exists():
        return '', 404
    from flask import send_file
    return send_file(str(img_path), mimetype='image/jpeg')


# ── Dashboard de progreso ────────────────────────────────────────────────────
@app.route('/dashboard')
def dashboard():
    countries = get_available_countries()
    return render_template('dashboard.html',
                           countries=countries,
                           current_country=current_country,
                           current_user=get_current_user())


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
        country   = current_country
        available = set(enc_class for enc_class in le.classes_)

    words = [w for w in SPELL_WORDS.get(country, SPELL_WORDS['asl'])
             if all(c in available for c in w)]

    return render_template('spell.html',
                           countries=countries,
                           current_country=country,
                           words=words,
                           current_user=get_current_user())


# ── API: historial de sesiones (MongoDB) ─────────────────────────────────────
@app.route('/api/session', methods=['POST'])
def api_save_session():
    user = get_current_user()
    if not user:
        return jsonify({'ok': False, 'error': 'No autenticado'}), 401
    doc = request.get_json(silent=True)
    if not doc:
        return jsonify({'ok': False, 'error': 'Sin datos'}), 400
    save_session(user['id'], doc)
    return jsonify({'ok': True})


@app.route('/api/history', methods=['GET'])
def api_history():
    user = get_current_user()
    if not user:
        return jsonify({'ok': False, 'error': 'No autenticado'}), 401
    country = request.args.get('country')
    limit   = int(request.args.get('limit', 200))
    data    = get_history(user['id'], country=country, limit=limit)
    return jsonify({'ok': True, **data})


@app.route('/api/history', methods=['DELETE'])
def api_delete_history():
    user = get_current_user()
    if not user:
        return jsonify({'ok': False, 'error': 'No autenticado'}), 401
    country = request.args.get('country')
    count   = delete_history(user['id'], country=country)
    return jsonify({'ok': True, 'deleted': count})


# ── Captura de datos de entrenamiento (web) ──────────────────────────────────
_capture_locks = {c: threading.Lock() for c in ('asl', 'colombia', 'china')}

@app.route('/capture')
def capture_page():
    countries = get_available_countries()
    return render_template('capture.html',
                           countries=countries,
                           current_country=current_country,
                           current_user=get_current_user())


@app.route('/training')
def training_studio():
    if flask_session.get('email') != TRAINING_ALLOWED_EMAIL:
        return render_template('403.html', current_user=get_current_user()), 403
    countries = get_available_countries()
    return render_template('training_studio.html',
                           countries=countries,
                           current_country=current_country,
                           current_user=get_current_user())


@app.route('/capture/save', methods=['POST'])
def capture_save():
    from config import LANDMARK_COLUMNS
    data    = request.get_json(silent=True) or {}
    frame_b64 = data.get('frame', '')
    label     = data.get('label', '').upper()
    country   = data.get('country', 'asl')

    if not label or country not in COUNTRIES:
        return jsonify({'ok': False})

    try:
        frame_bytes = base64.b64decode(frame_b64)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return jsonify({'ok': False, 'detected': False})

    if frame is None:
        return jsonify({'ok': False, 'detected': False})

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    lm, _ = extract_landmarks(rgb)

    if lm is None:
        return jsonify({'ok': True, 'detected': False, 'saved': False})

    folder = 'China' if country == 'china' else country
    data_dir = BASE_DIR / 'data' / folder
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / 'landmarks.csv'

    with _capture_locks[country]:
        file_exists = csv_path.exists()
        with open(csv_path, 'a', newline='') as f:
            writer = csv_module.DictWriter(f, fieldnames=LANDMARK_COLUMNS)
            if not file_exists:
                writer.writeheader()
            row = {f'x{i}': lm[i*3] for i in range(21)}
            row.update({f'y{i}': lm[i*3+1] for i in range(21)})
            row.update({f'z{i}': lm[i*3+2] for i in range(21)})
            row['label'] = label
            writer.writerow(row)

    return jsonify({'ok': True, 'detected': True, 'saved': True})


@app.route('/capture/reset_letter', methods=['POST'])
def capture_reset_letter():
    import pandas as pd
    data    = request.get_json(silent=True) or {}
    label   = data.get('label', '').upper()
    country = data.get('country', 'asl')

    if not label or country not in COUNTRIES:
        return jsonify({'ok': False, 'error': 'Datos inválidos'}), 400

    folder   = 'China' if country == 'china' else country
    csv_path = BASE_DIR / 'data' / folder / 'landmarks.csv'

    if not csv_path.exists():
        return jsonify({'ok': True, 'deleted': 0})

    with _capture_locks[country]:
        df      = pd.read_csv(csv_path)
        deleted = int((df['label'] == label).sum())
        df      = df[df['label'] != label]
        if df.empty:
            csv_path.unlink()
        else:
            df.to_csv(csv_path, index=False)

    return jsonify({'ok': True, 'deleted': deleted})


@app.route('/capture/counts')
def capture_counts():
    country  = request.args.get('country', 'asl')
    folder   = 'China' if country == 'china' else country
    csv_path = BASE_DIR / 'data' / folder / 'landmarks.csv'
    counts   = {}
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        counts = df['label'].value_counts().to_dict()
    return jsonify({'ok': True, 'counts': counts})


# ── Arranque ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import socket
    def _lan_ip():
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(('8.8.8.8', 80))
                return s.getsockname()[0]
        except OSError:
            return '127.0.0.1'

    port = int(os.environ.get('PORT', 5000))
    print(f"\n  Local:   http://127.0.0.1:{port}")
    print(f"  Network: http://{_lan_ip()}:{port}\n")
    app.run(host='0.0.0.0', port=port, debug=False)
