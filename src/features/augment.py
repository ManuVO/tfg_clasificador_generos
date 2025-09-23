"""Augmentations for waveform and spectrogram domains."""

from __future__ import annotations

import random
from typing import Dict, Optional

import numpy as np
from audiomentations import (
    Compose,
    AddGaussianNoise,
    TimeStretch,
    PitchShift,
    Shift,
    Gain,
)

WaveformAugmentConfig = Dict[str, Dict[str, float]]


def build_waveform_augment(config: Dict) -> Optional[Compose]:
    """Create a waveform augmentation pipeline from a configuration dictionary."""

    waveform_cfg: WaveformAugmentConfig = (
        config.get("augmentation", {}).get("waveform", {}) if config else {}
    )

    transforms = []

    registry = {
        "AddGaussianNoise": AddGaussianNoise,
        "TimeStretch": TimeStretch,
        "PitchShift": PitchShift,
        "Shift": Shift,
        "Gain": Gain,
    }

    for name, aug_cls in registry.items():
        params = waveform_cfg.get(name)
        if params:
            transforms.append(aug_cls(**params))

    if not transforms:
        return None

    return Compose(transforms)


def apply_spec_augment(spec: np.ndarray, config: Dict) -> np.ndarray:
    """Apply SpecAugment-style masking over a log-mel spectrogram."""

    if spec is None:
        return spec

    spec_cfg = config.get("augmentation", {}).get("spec_augment", {}) if config else {}

    prob = float(spec_cfg.get("p", 0.0))
    if prob <= 0.0 or random.random() > prob:
        return spec

    augmented = spec.copy()
    freq_masks = int(spec_cfg.get("frequency_masks", 0))
    freq_max = int(spec_cfg.get("frequency_max_width", 0))
    time_masks = int(spec_cfg.get("time_masks", 0))
    time_max = int(spec_cfg.get("time_max_width", 0))

    num_mels, num_frames = augmented.shape
    replacement_value = float(np.min(augmented))

    for _ in range(freq_masks):
        width = random.randint(0, max(freq_max, 0)) if freq_max > 0 else 0
        if width == 0:
            continue
        start = random.randint(0, max(num_mels - width, 0))
        augmented[start : start + width, :] = replacement_value

    for _ in range(time_masks):
        width = random.randint(0, max(time_max, 0)) if time_max > 0 else 0
        if width == 0:
            continue
        start = random.randint(0, max(num_frames - width, 0))
        augmented[:, start : start + width] = replacement_value

    return augmented


__all__ = ["build_waveform_augment", "apply_spec_augment"]
