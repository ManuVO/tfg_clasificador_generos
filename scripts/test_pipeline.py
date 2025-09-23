# scripts/test_pipeline.py

from pathlib import Path

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

from features.augment import build_waveform_augment, apply_spec_augment
from features.melspectrogram import mel_spec
from configuration import load_stage_config

config, config_meta = load_stage_config("training")
dataset_name = config_meta.get("dataset", "desconocido")
print(f"⚙️  Configuración de prueba cargada para dataset={dataset_name}.")

TMP = Path("reports") / "tmp_pipeline"
TMP.mkdir(parents=True, exist_ok=True)

audio_cfg = config.get("audio", {})
sr = audio_cfg.get("sample_rate", 22050)
hop_length = audio_cfg.get("hop_length", 512)
n_mels = audio_cfg.get("n_mels", 128)

print(f"Parámetros de prueba: sr={sr}, hop_length={hop_length}")

# 1. Generar o cargar audio de prueba
# Usaremos un tono sintético para tener una entrada controlada.
duration = 3.0  # 3 segundos de audio de prueba
freq = 440.0    # tono de 440 Hz (La4)
t = np.linspace(0, duration, int(sr * duration), endpoint=False)
y_orig = 0.2 * np.sin(2 * np.pi * freq * t).astype(np.float32)  # onda senoidal a 0.2 de amplitud

waveform_aug = build_waveform_augment(config)
if waveform_aug:
    y_aug = waveform_aug(samples=y_orig, sample_rate=sr)
    print(
        "🟡 Augmentations waveforms activas",
        f"| longitud original = {len(y_orig)} | tras aug = {len(y_aug)}",
    )
    diff = np.linalg.norm(y_orig - y_aug)
    print(f"   -> Diferencia L2 entre señal original y augmentada: {diff:.2f}")
else:
    y_aug = y_orig
    print("🟡 No hay augmentations de waveform configuradas.")

# 3. Calcular mel-espectrogramas para verificar hop_length y dimensiones
S_orig = mel_spec(y_orig, sr=sr, hop_length=hop_length, n_mels=n_mels)
S_aug = mel_spec(y_aug, sr=sr, hop_length=hop_length, n_mels=n_mels)
S_aug = apply_spec_augment(S_aug, config)
print(f"🟢 Mel-espectrograma original: shape = {S_orig.shape} (mel_bins, frames)")
print(f"🟢 Mel-espectrograma augmentado: shape = {S_aug.shape} (mel_bins, frames)")

# Verificar dimensiones esperadas en función de hop_length
if S_orig.shape[0] == n_mels:
    print(f"   -> Número de filtros Mel correcto ({S_orig.shape[0]}).")
else:
    print(f"   -> ❌ Número de filtros Mel inesperado: {S_orig.shape[0]} (esperado {n_mels}).")

# Comparar número de frames de espectrograma original vs augmentado
frames_orig = S_orig.shape[1]
frames_aug = S_aug.shape[1]
print(f"   -> Frames espectrograma original: {frames_orig} | tras aug: {frames_aug}")
if frames_orig != frames_aug:
    print("   -> La cantidad de frames difiere tras augmentación (TimeStretch/Shift modificó la duración).")

# 4. Visualizar opcionalmente el mel-espectrograma para inspección
plt.figure(figsize=(6, 4))
librosa.display.specshow(S_aug, sr=sr, hop_length=hop_length,
                         x_axis="time", y_axis="mel", fmax=8000)
plt.colorbar(format="%+2.0f dB")
plt.title("Mel-espectrograma (señal augmentada)")
plt.tight_layout()
plt.savefig(TMP / "spec_example.png", dpi=120)
plt.close()
print("✅ Prueba de pipeline completada.")
