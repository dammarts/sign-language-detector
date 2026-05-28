"""
Uso:
    python model/train.py                    # entrena con datos ASL base
    python model/train.py --country colombia # entrena modelo Colombia
    python model/train.py --country china    # entrena modelo China
"""
import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import COUNTRIES, DATA_SOURCES, MODEL_BASE


def train(country: str):
    if country not in COUNTRIES:
        raise ValueError(f"País '{country}' no registrado. Opciones: {list(COUNTRIES.keys())}")

    print(f"\n=== Entrenando modelo: {COUNTRIES[country]['name']} ===\n")

    dfs = []
    for path in DATA_SOURCES[country]:
        if path.exists():
            df_part = pd.read_csv(path)
            print(f"  Cargado: {path.name} -> {len(df_part)} filas")
            dfs.append(df_part)
        else:
            print(f"  Saltando (no existe): {path}")

    if not dfs:
        raise FileNotFoundError(
            f"No se encontraron datos para '{country}'.\n"
            f"Captura datos con: python capture_data.py --country {country}"
        )

    df = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total filas: {len(df)}")

    X = df.drop(columns=['label']).values
    y = df['label'].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f"\n  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    out_dir = MODEL_BASE / country
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / 'classifier.pkl', 'wb') as f:
        pickle.dump(clf, f)
    with open(out_dir / 'label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    print(f"  Modelo guardado en: {out_dir}/\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entrenar modelo de lenguaje de señas')
    parser.add_argument('--country', default='asl',
                        choices=list(COUNTRIES.keys()),
                        help='País / lengua de señas a entrenar')
    args = parser.parse_args()
    train(args.country)
