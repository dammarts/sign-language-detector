from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ── Países soportados ────────────────────────────────────────────────────────
COUNTRIES = {
    'asl':      {'name': 'ASL (Estados Unidos)', 'flag': '🇺🇸'},
    'colombia': {'name': 'Colombia',              'flag': '🇨🇴'},
    'china':    {'name': 'China',                 'flag': 'Ch'},
}

# ── Fuentes de datos por país (para entrenamiento) ───────────────────────────
DATA_SOURCES = {
    'asl': [
        BASE_DIR / 'data' / 'archive' / 'asl_landmarks_final.csv',
        BASE_DIR / 'data' / 'asl' / 'landmarks.csv',
    ],
    'colombia': [
        BASE_DIR / 'data' / 'colombia' / 'landmarks.csv',
    ],
    'china': [
        BASE_DIR / 'data' / 'China' / 'landmarks.csv',
    ],
}

# ── Captura de datos ─────────────────────────────────────────────────────────
SAMPLES_PER_KEY = 100
CAPTURE_LABELS  = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['del', 'space']

# ── Esquema CSV (21 landmarks × x,y,z + etiqueta) ───────────────────────────
LANDMARK_COLUMNS = [f'{a}{i}' for i in range(21) for a in ['x', 'y', 'z']] + ['label']

# ── Rutas de modelos ─────────────────────────────────────────────────────────
MODEL_BASE = BASE_DIR / 'model'
