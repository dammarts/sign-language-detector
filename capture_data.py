"""
Uso:
    python capture_data.py                    # captura para ASL (por defecto)
    python capture_data.py --country colombia # captura para Colombia
    python capture_data.py --country china    # captura para China

Controles:
    A-Z  → selecciona la letra a capturar
    ESC  → salir
"""
import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

import cv2

from config import (CAPTURE_LABELS, COUNTRIES, LANDMARK_COLUMNS,
                    MODEL_BASE, SAMPLES_PER_KEY)
from utils.hand_detector import extract_landmarks, mp_drawing, mp_hands

SAMPLES_TARGET = 300  # muestras por letra en esta sesión


def load_existing_counts(csv_path):
    counts = Counter()
    if not os.path.isfile(csv_path):
        return counts
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            counts[row.get('label', '')] += 1
    return counts


def load_weak_letters(country):
    metrics_path = MODEL_BASE / country / 'metrics.json'
    weak = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            data = json.load(f)
        for k, v in data.get('per_class', {}).items():
            if len(k) == 1 and k.isalpha():
                weak[k] = v.get('precision', 1.0)
    return weak


def print_summary(counts, weak, country_name):
    print(f"\n  === {country_name} — Estado del dataset ===\n")
    alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    for i, letter in enumerate(alphabet):
        n = counts.get(letter, 0)
        prec = weak.get(letter)
        bar = '█' * min(n // 20, 20)
        tag = ''
        if prec is not None and prec < 0.90:
            tag = f'  <-- DEBIL ({prec:.0%})'
        print(f'  {letter}: {n:>5} muestras  {bar}{tag}')
    print()


def run(country: str):
    if country not in COUNTRIES:
        raise ValueError(f"Pais '{country}' no valido. Opciones: {list(COUNTRIES.keys())}")

    output_dir = os.path.join('data', country if country != 'china' else 'China')
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, 'landmarks.csv')

    counts = load_existing_counts(output_csv)
    weak   = load_weak_letters(country)
    country_name = COUNTRIES[country]['name']

    print_summary(counts, weak, country_name)
    if weak:
        weak_list = [k for k, v in sorted(weak.items(), key=lambda x: x[1]) if v < 0.90]
        if weak_list:
            print(f"  Letras prioritarias: {', '.join(weak_list)}\n")

    print("  Instrucciones:")
    print("    Presiona A-Z para capturar esa seña")
    print(f"    Se capturan {SAMPLES_TARGET} muestras por letra")
    print("    Presiona ESC para salir\n")

    file_exists = os.path.isfile(output_csv)
    csv_file = open(output_csv, 'a', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=LANDMARK_COLUMNS)
    if not file_exists:
        writer.writeheader()

    cap = cv2.VideoCapture(0)
    current_label = None
    session_count = 0  # muestras capturadas en esta sesión para la letra actual

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks, hand_lm = extract_landmarks(frame_rgb)

        if hand_lm is not None:
            mp_drawing.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

            if current_label is not None and session_count < SAMPLES_TARGET and landmarks is not None:
                row = {}
                for i in range(21):
                    base = i * 3
                    row[f'x{i}'] = landmarks[base]
                    row[f'y{i}'] = landmarks[base + 1]
                    row[f'z{i}'] = landmarks[base + 2]
                row['label'] = current_label
                writer.writerow(row)
                csv_file.flush()
                session_count += 1
                counts[current_label] = counts.get(current_label, 0) + 1

        # ── HUD ──────────────────────────────────────────────────────────────
        cv2.putText(frame, f'Pais: {country_name}', (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        if current_label:
            total = counts.get(current_label, 0)
            prec  = weak.get(current_label)
            prec_txt = f' | modelo: {prec:.0%}' if prec is not None else ''
            status = f'Letra: {current_label}  sesion: {session_count}/{SAMPLES_TARGET}  total: {total}{prec_txt}'
            color  = (0, 255, 0) if hand_lm is not None else (0, 0, 255)
            cv2.putText(frame, status, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            if session_count >= SAMPLES_TARGET:
                cv2.putText(frame, 'Listo! Presiona otra letra', (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        else:
            cv2.putText(frame, 'Presiona una letra para empezar', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2)

        cv2.imshow(f'Captura: {country_name}', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break
        elif key != 255:
            char = chr(key).upper()
            if char in CAPTURE_LABELS:
                current_label = char
                session_count = 0
                total = counts.get(char, 0)
                prec  = weak.get(char)
                prec_txt = f' (modelo: {prec:.0%})' if prec is not None else ''
                print(f'  Capturando: {char}  | ya tienes {total} muestras{prec_txt}')

    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()

    print(f"\n  Datos guardados en {output_csv}")
    print(f"  Total muestras por letra (post-sesion):")
    for letter in sorted(counts):
        if len(letter) == 1 and letter.isalpha():
            print(f'    {letter}: {counts[letter]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Capturar landmarks de lenguaje de senas')
    parser.add_argument('--country', default='asl',
                        choices=list(COUNTRIES.keys()),
                        help='Pais / lengua de senas a capturar')
    args = parser.parse_args()
    run(args.country)
