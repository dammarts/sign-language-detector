"""
Uso:
    python capture_data.py                    # captura para ASL (por defecto)
    python capture_data.py --country colombia # captura para Colombia
    python capture_data.py --country china    # captura para China

Controles:
    A-Z        → selecciona la letra a capturar
    ESC        → salir
"""
import argparse
import csv
import os

import cv2

from config import COUNTRIES, CAPTURE_LABELS, LANDMARK_COLUMNS, SAMPLES_PER_KEY
from utils.hand_detector import extract_landmarks, mp_drawing, mp_hands


def run(country: str):
    if country not in COUNTRIES:
        raise ValueError(f"País '{country}' no válido. Opciones: {list(COUNTRIES.keys())}")

    output_dir = os.path.join('data', country)
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, 'landmarks.csv')

    file_exists = os.path.isfile(output_csv)
    csv_file = open(output_csv, 'a', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=LANDMARK_COLUMNS)
    if not file_exists:
        writer.writeheader()

    cap = cv2.VideoCapture(0)
    current_label = None
    count = 0

    country_name = COUNTRIES[country]['name']
    print(f"\n  País: {country_name}")
    print(f"  Guardando en: {output_csv}")
    print("\n  Instrucciones:")
    print("    Presiona A-Z para capturar esa seña")
    print("    Presiona ESC para salir\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks, hand_lm = extract_landmarks(frame_rgb)

        if hand_lm is not None:
            mp_drawing.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

            if current_label is not None and count < SAMPLES_PER_KEY and landmarks is not None:
                row = {}
                for i in range(21):
                    base = i * 3
                    row[f'x{i}'] = landmarks[base]
                    row[f'y{i}'] = landmarks[base + 1]
                    row[f'z{i}'] = landmarks[base + 2]
                row['label'] = current_label
                writer.writerow(row)
                csv_file.flush()
                count += 1

        cv2.putText(frame, f'Pais: {country_name}', (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)

        if current_label:
            status = f'Letra: {current_label} | {count}/{SAMPLES_PER_KEY}'
            color = (0, 255, 0) if hand_lm is not None else (0, 0, 255)
            cv2.putText(frame, status, (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if count >= SAMPLES_PER_KEY:
                cv2.putText(frame, 'Listo! Presiona otra letra', (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            cv2.putText(frame, 'Presiona una letra para empezar', (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

        cv2.imshow(f'Captura: {country_name}', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break
        elif key != 255:
            char = chr(key).upper()
            if char in CAPTURE_LABELS:
                current_label = char
                count = 0
                print(f"  Capturando: {char}")

    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n  Datos guardados en {output_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Capturar landmarks de lenguaje de señas')
    parser.add_argument('--country', default='asl',
                        choices=list(COUNTRIES.keys()),
                        help='País / lengua de señas a capturar')
    args = parser.parse_args()
    run(args.country)
