# src/data/dataset.py (después de los cambios)
import yaml
import soundfile as sf
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

from features.melspectrogram import mel_spec
from features.augment import train_aug

# Cargar configuración YAML
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
HOP_LENGTH = config["audio"]["hop_length"]

class GenreDataset(Dataset):
    def __init__(self, items, split, sr=22050, slice_sec=3, augment=False):
        self.items = [row for row in items if row["split"] == split]
        self.sr = sr
        self.slice_sec = slice_sec
        self.augment = augment

    def __getitem__(self, idx):
        path = self.items[idx]["filepath"]
        y, sr = sf.read(path)
        if self.augment:
            y = train_aug(samples=y, sample_rate=sr)
        # Usar hop_length desde la configuración en el mel_spec
        S = mel_spec(y, sr, hop_length=HOP_LENGTH)
        S = (S - S.mean()) / (S.std() + 1e-6)
        x = torch.tensor(S).unsqueeze(0)
        ylab = torch.tensor(self.items[idx]["label_idx"])
        return x, ylab


# import torch
# from torch.utils.data import Dataset
# import soundfile as sf
# import numpy as np
# import librosa

# from features.melspectrogram import mel_spec
# from features.augment import train_aug

# class GenreDataset(Dataset):
#     def __init__(self, items, split, sr=22050, slice_sec=3, augment=False):
#         self.items = [row for row in items if row["split"]==split]
#         self.sr = sr; self.slice_sec = slice_sec; self.augment = augment

#     def __getitem__(self, idx):
#         path = self.items[idx]["filepath"]
#         y, sr = sf.read(path)
#         if self.augment:
#             y = train_aug(samples=y, sample_rate=sr)
#         S = mel_spec(y, sr)
#         S = (S - S.mean()) / (S.std() + 1e-6)
#         x = torch.tensor(S).unsqueeze(0)
#         ylab = torch.tensor(self.items[idx]["label_idx"])
#         return x, ylab

#     def __len__(self):
#         return len(self.items)