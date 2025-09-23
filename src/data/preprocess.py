"""Dataset preprocessing utilities for GTZAN."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import librosa
import numpy as np
import soundfile as sf
import yaml
from pydub import AudioSegment
from sklearn.model_selection import train_test_split

RNG_SEED = 42
random.seed(RNG_SEED)

RAW_DIR = Path("data/raw/gtzan")
PROC_DIR = Path("data/processed/gtzan")
PROC_AUDIO_DIR = PROC_DIR / "segments"
CSV_PATH = PROC_DIR / "metadata.csv"

GENRE_DIR_CANDIDATES = [
    RAW_DIR / "Data" / "genres_original",
    RAW_DIR / "genres_original",
    RAW_DIR / "genres",
]


@dataclass
class RunningStats:
    count: int = 0
    sum_: float = 0.0
    sumsq: float = 0.0

    def update(self, array: np.ndarray) -> None:
        arr = np.asarray(array, dtype=np.float64)
        self.count += arr.size
        self.sum_ += float(arr.sum())
        self.sumsq += float(np.square(arr).sum())

    def to_dict(self) -> Dict[str, float]:
        if self.count == 0:
            return {"mean": 0.0, "std": 1.0, "count": 0}
        mean = self.sum_ / self.count
        variance = max(self.sumsq / self.count - mean**2, 0.0)
        std = float(np.sqrt(variance))
        return {"mean": float(mean), "std": std, "count": int(self.count)}


def find_genres_root() -> Path:
    for path in GENRE_DIR_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No se encontró la carpeta de géneros."
    )


def standardize_mono_resample(y: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    maxv = np.max(np.abs(y)) + 1e-9
    return (y / maxv).astype(np.float32)


def slice_signal(
    y: np.ndarray, sr: int, slice_sec: float, hop_sec: float
) -> Iterator[Tuple[np.ndarray, int, int]]:
    n = int(slice_sec * sr)
    hop = int(hop_sec * sr)
    if len(y) < n:
        return
    for start in range(0, len(y) - n + 1, hop):
        end = start + n
        yield y[start:end], start, end


def safe_load_audio(path: Path, target_sr: int, mono: bool = True):
    try:
        y, sr = librosa.load(path, sr=None, mono=mono)
        return y, sr
    except Exception as exc:
        print(f"[WARN] librosa no pudo abrir {path}: {exc}. Intentando con pydub/ffmpeg…")
        try:
            audio = AudioSegment.from_file(path)
            if mono and audio.channels > 1:
                audio = audio.set_channels(1)
            if audio.frame_rate != target_sr:
                audio = audio.set_frame_rate(target_sr)
            raw = audio.raw_data
            waveform = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            return waveform, audio.frame_rate
        except Exception as exc2:
            print(
                f"[ERROR] pydub/ffmpeg tampoco pudo abrir {path}: {exc2}. Se omitirá este archivo."
            )
            return None, None


def mel_spectrogram_db(
    waveform: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=20,
        fmax=8000,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)


def prepare(config_path: Path | str = "configs/gtzan_cnn.yaml") -> None:
    with open(config_path, "r", encoding="utf-8") as cfg_file:
        config = yaml.safe_load(cfg_file)

    audio_cfg = config.get("audio", {})
    target_sr = int(audio_cfg.get("sample_rate", 22050))
    slice_sec = float(audio_cfg.get("slice_duration", 3.0))
    hop_sec = slice_sec / 2
    mel_n_fft = int(audio_cfg.get("n_fft", 2048))
    mel_hop = int(audio_cfg.get("hop_length", 512))
    mel_bins = int(audio_cfg.get("n_mels", 128))

    genres_root = find_genres_root()
    PROC_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    tracks: List[Dict] = []
    genres = sorted([d.name for d in genres_root.iterdir() if d.is_dir()])
    label2idx = {genre: idx for idx, genre in enumerate(genres)}

    print("-> Escaneando archivos…")
    for genre in genres:
        for wav in (genres_root / genre).glob("*.wav"):
            tracks.append({"filepath": wav, "genre": genre, "label_idx": label2idx[genre]})
        for au in (genres_root / genre).glob("*.au"):
            tracks.append({"filepath": au, "genre": genre, "label_idx": label2idx[genre]})

    print(f"Encontradas {len(tracks)} pistas en {len(genres)} géneros.")

    paths = [str(track["filepath"]) for track in tracks]
    labels = [track["label_idx"] for track in tracks]

    p_train = 0.8
    p_val = 0.1

    train_paths, tmp_paths, train_labels, tmp_labels = train_test_split(
        paths, labels, test_size=1 - p_train, stratify=labels, random_state=RNG_SEED
    )
    val_paths, test_paths, _, _ = train_test_split(
        tmp_paths,
        tmp_labels,
        test_size=(1 - p_train - p_val) / (1 - p_train),
        stratify=tmp_labels,
        random_state=RNG_SEED,
    )

    split_by_path = {path: "train" for path in train_paths}
    split_by_path.update({path: "val" for path in val_paths})
    split_by_path.update({path: "test" for path in test_paths})

    rows: List[Dict] = []
    seg_count = 0
    train_stats = RunningStats()

    print("-> Procesando y troceando… (puede tardar unos minutos)")

    for track in tracks:
        src = Path(track["filepath"])
        split = split_by_path[str(src)]
        waveform, sr = safe_load_audio(src, target_sr=target_sr, mono=True)
        if waveform is None:
            continue

        waveform = standardize_mono_resample(waveform, sr, target_sr)

        for seg_idx, (segment, start, end) in enumerate(
            slice_signal(waveform, target_sr, slice_sec, hop_sec)
        ):
            out_dir = PROC_AUDIO_DIR / split / track["genre"]
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{src.stem}_seg{seg_idx:04d}.wav"
            sf.write(out_path, segment, target_sr)

            duration_sec = len(segment) / target_sr
            start_sec = start / target_sr
            end_sec = end / target_sr

            rows.append(
                {
                    "filepath": str(out_path.as_posix()),
                    "label": track["genre"],
                    "label_idx": track["label_idx"],
                    "split": split,
                    "track_path": str(src.as_posix()),
                    "segment_index": seg_idx,
                    "start_sec": round(start_sec, 3),
                    "end_sec": round(end_sec, 3),
                    "duration_sec": round(duration_sec, 3),
                }
            )

            if split == "train":
                mel_db = mel_spectrogram_db(segment, target_sr, mel_n_fft, mel_hop, mel_bins)
                train_stats.update(mel_db)

            seg_count += 1

    PROC_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "filepath",
        "label",
        "label_idx",
        "split",
        "track_path",
        "segment_index",
        "start_sec",
        "end_sec",
        "duration_sec",
    ]

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    norm_stats_path = Path(config["dataset"]["norm_stats_path"])
    if not norm_stats_path.is_absolute():
        norm_stats_path = Path.cwd() / norm_stats_path

    norm_stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(norm_stats_path, "w", encoding="utf-8") as stats_file:
        json.dump(train_stats.to_dict(), stats_file, indent=2)

    print(f"✅ Hecho. Segmentos guardados: {seg_count}")
    print(f"CSV: {CSV_PATH.resolve()}")
    print(f"Estadísticas de normalización: {norm_stats_path}")


if __name__ == "__main__":
    prepare()
