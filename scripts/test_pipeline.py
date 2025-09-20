# scripts/test_pipeline.py

import os
import numpy as np
import yaml
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Importar nuestras funciones del proyecto
from features.augment import train_aug
from features.melspectrogram import mel_spec

# Cargar la configuración (config.yaml) para obtener parámetros
config_path = os.path.join("configs", "gtzan_cnn.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

sr = config.get("sr", 22050)  # frecuencia de muestreo esperada (ej: 22050 Hz)
hop_length = config.get("hop_length", 512)  # hop_length definido en config (ej: 512):contentReference[oaicite:6]{index=6}

print(f"Parámetros de prueba: sr={sr}, hop_length={hop_length}")

# 1. Generar o cargar audio de prueba
# Usaremos un tono sintético para tener una entrada controlada.
duration = 3.0  # 3 segundos de audio de prueba
freq = 440.0    # tono de 440 Hz (La4)
t = np.linspace(0, duration, int(sr * duration), endpoint=False)
y_orig = 0.2 * np.sin(2 * np.pi * freq * t).astype(np.float32)  # onda senoidal a 0.2 de amplitud

# 2. Aplicar augmentations de entrenamiento
# (Verificamos que solo se apliquen si augment=True en config)
if config.get("augment", False):
    # Aplicar la composición de transforms de audiomentations
    y_aug = train_aug(samples=y_orig, sample_rate=sr)
    print(f"🟡 Augmentations aplicadas: longitud original = {len(y_orig)} muestras, "
          f"longitud tras aug = {len(y_aug)} muestras")
    # Chequear si hubo cambio en longitud o contenido
    if len(y_orig) != len(y_aug):
        print("   -> La señal augmentada tiene distinta duración (p.ej., por TimeStretch).")
    # Comparar contenido (norma de diferencia) para ver si cambió
    diff = np.linalg.norm(y_orig - y_aug) 
    if diff == 0:
        print("   -> AVISO: Las augmentations no modificaron la señal (posible, dada la aleatoriedad).")
    else:
        print(f"   -> Diferencia entre señal original y augmentada (norma L2): {diff:.2f}")
else:
    y_aug = y_orig
    print("🟡 'augment' está deshabilitado en config; se omite la aplicación de augmentations.")

# 3. Calcular mel-espectrogramas para verificar hop_length y dimensiones
S_orig = mel_spec(y_orig, sr=sr, hop_length=hop_length)        # espectrograma de Mel de la señal original
S_aug  = mel_spec(y_aug,  sr=sr, hop_length=hop_length)        # espectrograma de Mel de la señal augmentada
print(f"🟢 Mel-espectrograma original: shape = {S_orig.shape} (mel_bins, frames)")
print(f"🟢 Mel-espectrograma augmentado: shape = {S_aug.shape} (mel_bins, frames)")

# Verificar dimensiones esperadas en función de hop_length
n_mels = config.get("n_mels", 128)
expected_mels = n_mels
if S_orig.shape[0] == expected_mels:
    print(f"   -> Número de filtros Mel correcto ({S_orig.shape[0]}).")
else:
    print(f"   -> ❌ Número de filtros Mel inesperado: {S_orig.shape[0]} (esperado {expected_mels}).")

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
plt.show()  # Mostrar la gráfica del espectrograma augmentado
print("✅ Prueba de pipeline completada.")