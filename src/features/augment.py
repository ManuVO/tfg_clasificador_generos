from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

train_aug = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.3),
    TimeStretch(min_rate=0.9, max_rate=1.1, p=0.3),
    PitchShift(min_semitones=-2, max_semitones=2, p=0.3),
    Shift(min_shift=-0.2, max_shift=0.2, p=0.2),
])
