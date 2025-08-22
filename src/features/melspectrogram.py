import librosa
import numpy as np

def mel_spec(y, sr, n_fft=2048, hop_length=512, n_mels=128):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=20, fmax=8000
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db.astype(np.float32)
