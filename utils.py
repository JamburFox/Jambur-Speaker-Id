import librosa
import numpy as np
import torch

# Function to extract MFCC features from an audio file
def extract_audio_features(audio: np.ndarray, sr: float, n_mfcc: int=13, hop_length: int=512, n_fft: int=2048) -> np.ndarray:
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
    mfccs = librosa.util.normalize(mfccs, axis=1)

    #Transpose to match input shape (time steps, features)
    mfccs = mfccs.T
    return mfccs

def load_audio(file_path: str) -> tuple[np.ndarray, int]:
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr

def load_audio_features(file_path: str) -> torch.Tensor:
    audio, sr = load_audio(file_path)
    audio_features = extract_audio_features(audio, sr)
    audio_features = torch.from_numpy(audio_features).unsqueeze(dim=0)
    return audio_features