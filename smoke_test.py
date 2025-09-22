import os, sys, math
from pathlib import Path
import numpy as np

# Asegurar que podemos importar src.*
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from models.cnn_baseline import CNNBaseline
from features.melspectrogram import mel_spec
from data.dataset import GenreDataset

# ------------- PRUEBA 0: imports de librerías base -------------
print("-> Comprobando imports...")
import torch
import librosa
import soundfile as sf
from sklearn.metrics import accuracy_score

print("OK imports | torch:", torch.__version__, "| librosa:", librosa.__version__)

# ------------- PRUEBA 1: forward del modelo con tensores random -------------


print("-> Forward aleatorio del modelo...")
model = CNNBaseline(n_classes=10)
x = torch.randn(4, 1, 128, 200)  # batch=4, 128 mel-bins, 200 frames
with torch.no_grad():
    y = model(x)
print("Salida shape:", tuple(y.shape))  # Esperado: (4, 10)
print("OK forward aleatorio")

# ------------- PRUEBA 2: pipeline de audio sintético -> mel_spec -> Dataset -------------
print("-> Generando WAVs sintéticos y construyendo Dataset...")

TMP = ROOT / "data" / "tmp_smoke"
TMP.mkdir(parents=True, exist_ok=True)
sr = 22050


def tone(freq, seconds=1.0, sr=22050, amp=0.2):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (amp * np.sin(2 * math.pi * freq * t)).astype(np.float32)


# Creamos 2 "clases" artificiales con tonos distintos
classes = ["clase_a", "clase_b"]
items = []
for i, (label, freq) in enumerate(zip(classes, [220, 440])):  # A3 vs A4
    for j in range(3):  # 3 wavs por clase
        y = tone(freq, seconds=1.0, sr=sr)
        wav_path = TMP / f"{label}_{j}.wav"
        sf.write(wav_path, y, sr)
        split = "train" if j < 2 else "val"
        items.append(
            {"filepath": str(wav_path), "label": label, "label_idx": i, "split": split}
        )

# Construimos datasets y comprobamos un batch
dummy_config = {
    "audio": {
        "sample_rate": sr,
        "slice_duration": 1.0,
        "n_fft": 2048,
        "hop_length": 512,
        "n_mels": 128,
    },
    "dataset": {"norm_stats_path": None},
    "augmentation": {
        "waveform": {},
        "spec_augment": {"p": 0.0},
    },
}

train_ds = GenreDataset(items, "train", config=dummy_config, augment=False)
val_ds = GenreDataset(items, "val", config=dummy_config, augment=False)

print(f"Train items: {len(train_ds)} | Val items: {len(val_ds)}")
x0, y0 = train_ds[0]
print("Ejemplo tensor:", tuple(x0.shape), "label_idx:", int(y0))  # (1, n_mels, T)
S = x0.squeeze(0).numpy()
print("Mel-spec ejemplo -> mel_bins:", S.shape[0], "frames:", S.shape[1])
print("OK Dataset y mel_spec")

# ------------- PRUEBA 3: mini entrenamiento (1 época, pasos cortos) -------------
print("-> Mini entrenamiento (1 época) para verificar loop...")

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

model = CNNBaseline(n_classes=len(classes))
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

train_dl = DataLoader(train_ds, batch_size=2, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=2, shuffle=False)

crit = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=1e-3)

# Entrenamiento muy corto
model.train()
for step, (xb, yb) in enumerate(train_dl):
    xb, yb = xb.to(device), yb.to(device)
    opt.zero_grad()
    logits = model(xb)
    loss = crit(logits, yb)
    loss.backward()
    opt.step()
    print(f"step {step} | loss {loss.item():.4f}")

# Validación mínima
model.eval()
preds, gts = [], []
with torch.no_grad():
    for xb, yb in val_dl:
        xb = xb.to(device)
        logits = model(xb)
        preds += logits.argmax(1).cpu().tolist()
        gts += yb.tolist()

acc = accuracy_score(gts, preds)
print("ACC valid (dummy):", acc)
print("OK mini entrenamiento")

print("\n✅ Smoke test completado con éxito.")
