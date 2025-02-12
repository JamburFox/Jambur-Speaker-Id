import librosa
import numpy as np

def load_audio(file_path: str):
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr

# Function to extract MFCC features from an audio file
def extract_audio_features(audio: np.ndarray, sr: float, n_mfcc=13, hop_length=512, n_fft=2048):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
    mfccs = librosa.util.normalize(mfccs, axis=1)

    #Transpose to match input shape (time steps, features)
    mfccs = mfccs.T
    return mfccs