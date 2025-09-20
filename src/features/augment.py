# src/features/augment.py (después de los cambios)
import yaml
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

# Cargar configuración YAML
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
aug_cfg = config["augmentation"]

# Definir las transformaciones de augmentación usando los parámetros del YAML
train_aug = Compose([
    AddGaussianNoise(**aug_cfg["AddGaussianNoise"]),
    TimeStretch(**aug_cfg["TimeStretch"]),
    PitchShift(**aug_cfg["PitchShift"]),
    Shift(**aug_cfg["Shift"]),
])


# from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

# train_aug = Compose([
#     AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.3),
#     TimeStretch(min_rate=0.9, max_rate=1.1, p=0.3),
#     PitchShift(min_semitones=-2, max_semitones=2, p=0.3),
#     Shift(min_shift=-0.2, max_shift=0.2, p=0.2),
# ])
