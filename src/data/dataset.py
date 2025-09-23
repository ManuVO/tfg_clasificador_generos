"""PyTorch Dataset implementation for the GTZAN segments."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from features.melspectrogram import mel_spec
from features.augment import build_waveform_augment, apply_spec_augment


class GenreDataset(Dataset):
    """Dataset of GTZAN segments returning log-mel spectrogram tensors."""

    def __init__(
        self,
        items: Iterable[Dict],
        split: str,
        config: Optional[Dict],
        augment: bool = False,
    ) -> None:
        self.items: List[Dict] = [row for row in items if row["split"] == split]
        self.split = split
        self.config = config or {}
        self.augment = augment

        audio_cfg = self.config.get("audio", {})
        self.sample_rate = int(audio_cfg.get("sample_rate", 22050))
        self.slice_duration = float(audio_cfg.get("slice_duration", 3.0))
        self.n_fft = int(audio_cfg.get("n_fft", 2048))
        self.hop_length = int(audio_cfg.get("hop_length", 512))
        self.n_mels = int(audio_cfg.get("n_mels", 128))

        self.waveform_aug = build_waveform_augment(self.config) if augment else None
        self.norm_stats = self._load_norm_stats()

    def _load_norm_stats(self) -> Optional[Dict[str, float]]:
        dataset_cfg = self.config.get("dataset", {})
        path = dataset_cfg.get("norm_stats_path")
        if not path:
            return None

        norm_path = Path(path)
        if not norm_path.is_absolute():
            norm_path = Path.cwd() / norm_path

        if not norm_path.exists():
            warnings.warn(
                f"Normalization stats file not found at {norm_path}. Falling back to per-segment standardization.",
                RuntimeWarning,
            )
            return None

        with open(norm_path, "r", encoding="utf-8") as f:
            stats = json.load(f)

        mean = float(stats.get("mean", 0.0))
        std = float(stats.get("std", 1.0))
        if std <= 0.0:
            std = 1.0

        return {"mean": mean, "std": std}

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        path = item["filepath"]

        waveform, sr = sf.read(path, dtype="float32")

        if self.waveform_aug is not None:
            waveform = self.waveform_aug(samples=waveform, sample_rate=sr)

        spec = mel_spec(
            waveform,
            sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

        if self.augment:
            spec = apply_spec_augment(spec, self.config)

        if self.norm_stats:
            spec = (spec - self.norm_stats["mean"]) / (self.norm_stats["std"] + 1e-8)
        else:
            spec = (spec - np.mean(spec)) / (np.std(spec) + 1e-6)

        spec_tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
        label_idx = torch.tensor(int(item["label_idx"]), dtype=torch.long)

        return spec_tensor, label_idx
