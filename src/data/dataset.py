import torch
from torch.utils.data import Dataset
import soundfile as sf
import numpy as np
import librosa

from features.melspectrogram import mel_spec
from features.augment import train_aug

class GenreDataset(Dataset):
    def __init__(self, items, split, sr=22050, slice_sec=3, augment=False):
        self.items = [row for row in items if row["split"]==split]
        self.sr = sr; self.slice_sec = slice_sec; self.augment = augment

    def __getitem__(self, idx):
        path = self.items[idx]["filepath"]
        y, sr = sf.read(path)
        if self.augment:
            y = train_aug(samples=y, sample_rate=sr)
        S = mel_spec(y, sr)
        S = (S - S.mean()) / (S.std() + 1e-6)
        x = torch.tensor(S).unsqueeze(0)
        ylab = torch.tensor(self.items[idx]["label_idx"])
        return x, ylab

    def __len__(self):
        return len(self.items)
