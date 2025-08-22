# src/data/preprocess.py
import csv
from pathlib import Path
from typing import List, Dict
import random

from pydub import AudioSegment
import io

import librosa
import soundfile as sf
import numpy as np
from sklearn.model_selection import train_test_split

RNG_SEED = 42
random.seed(RNG_SEED)

SR = 22050
SLICE_SEC = 3.0  # duración del segmento
HOP_SEC = 3.0  # sin solape (cámbialo a 1.5 para 50% solape si quieres)

RAW_DIR = Path("data/raw/gtzan")
PROC_DIR = Path("data/processed/gtzan")
PROC_AUDIO_DIR = PROC_DIR / "segments"
CSV_PATH = PROC_DIR / "metadata.csv"

GENRE_DIR_CANDIDATES = [
    RAW_DIR / "Data" / "genres_original",
    RAW_DIR / "genres_original",
    RAW_DIR / "genres",
]


def find_genres_root() -> Path:
    for p in GENRE_DIR_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        print("Directorio actual:", Path.cwd()),
        print("Ruta esperada data/raw/gtzan:", (Path("data/raw/gtzan").resolve())),
        f"No se encontró la carpeta de géneros. Busca en: {GENRE_DIR_CANDIDATES}",
    )


def standardize_mono_resample(
    y: np.ndarray, sr: int, target_sr: int = SR
) -> np.ndarray:
    # y viene mono si lo cargamos con librosa(mono=True)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    # normalización a -1..1 (librosa ya retorna float32 normalmente)
    maxv = np.max(np.abs(y)) + 1e-9
    y = (y / maxv).astype(np.float32)
    return y


def slice_signal(y: np.ndarray, sr: int, slice_sec: float, hop_sec: float):
    n = int(slice_sec * sr)
    h = int(hop_sec * sr)
    if len(y) < n:
        return
    for start in range(0, len(y) - n + 1, h):
        yield y[start : start + n]


def safe_load_audio(path, target_sr=SR, mono=True):
    """
    Intenta cargar audio con librosa; si falla, usa pydub+ffmpeg para decodificar
    a PCM y devolver un numpy. Devuelve (y, sr).
    """
    try:
        y, sr = librosa.load(path, sr=None, mono=mono)  # paso 1: soundfile/audioread
        # normalizamos SR fuera para mantener igual al flujo actual
        return y, sr
    except Exception as e1:
        print(f"[WARN] librosa no pudo abrir {path}: {e1}. Intento con pydub/ffmpeg...")
        try:
            # pydub lee casi cualquier cosa si FFmpeg está instalado
            audio = AudioSegment.from_file(path)
            if mono and audio.channels > 1:
                audio = audio.set_channels(1)
            if audio.frame_rate != target_sr:
                audio = audio.set_frame_rate(target_sr)
            # pydub -> bytes PCM int16 -> numpy float32 -1..1
            raw = audio.raw_data
            y = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            sr = audio.frame_rate
            return y, sr
        except Exception as e2:
            print(f"[ERROR] pydub/ffmpeg tampoco pudo abrir {path}: {e2}. Se omitirá este archivo.")
            return None, None


def prepare():
    genres_root = find_genres_root()
    PROC_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Construir lista de canciones por género
    tracks: List[Dict] = []
    genres = sorted([d.name for d in genres_root.iterdir() if d.is_dir()])
    label2idx = {g: i for i, g in enumerate(genres)}

    print("-> Escaneando archivos…")
    for g in genres:
        for wav in (genres_root / g).glob("*.wav"):
            tracks.append({"filepath": wav, "genre": g, "label_idx": label2idx[g]})
        for au in (genres_root / g).glob("*.au"):
            tracks.append({"filepath": au, "genre": g, "label_idx": label2idx[g]})

    print(f"Encontradas {len(tracks)} pistas en {len(genres)} géneros.")

    # 2) Split por CANCIÓN (evita fuga entre train/val/test)
    paths = [str(t["filepath"]) for t in tracks]
    labels = [t["label_idx"] for t in tracks]
    p_train = 0.8
    p_val = 0.1
    train_paths, tmp_paths, train_labels, tmp_labels = train_test_split(
        paths, labels, test_size=1 - p_train, stratify=labels, random_state=RNG_SEED
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        tmp_paths,
        tmp_labels,
        test_size=(1 - p_train - p_val) / (1 - p_train),
        stratify=tmp_labels,
        random_state=RNG_SEED,
    )

    split_by_path = {}
    for p in train_paths:
        split_by_path[p] = "train"
    for p in val_paths:
        split_by_path[p] = "val"
    for p in test_paths:
        split_by_path[p] = "test"

    # 3) Estandariza y trocea → guarda segmentos .wav + CSV
    rows = []
    seg_count = 0
    print("-> Procesando y troceando… (puede tardar unos minutos)")

    for t in tracks:
        src = Path(t["filepath"])
        split = split_by_path[str(src)]
        y, sr = safe_load_audio(src, target_sr=SR, mono=True)
        if y is None:
            continue
        y = standardize_mono_resample(y, sr, SR)

        for i, seg in enumerate(slice_signal(y, SR, SLICE_SEC, HOP_SEC)):
            out_dir = PROC_AUDIO_DIR / split / t["genre"]
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{src.stem}_seg{i:04d}.wav"
            sf.write(out_path, seg, SR)
            rows.append(
                {
                    "filepath": str(out_path.as_posix()),
                    "label": t["genre"],
                    "label_idx": t["label_idx"],
                    "split": split,
                }
            )
            seg_count += 1

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filepath", "label", "label_idx", "split"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Hecho. Segmentos guardados: {seg_count}")
    print(f"CSV: {CSV_PATH.resolve()}")
    print(f"Ejemplo de salida: {PROC_AUDIO_DIR.resolve()}")


if __name__ == "__main__":
    prepare()
