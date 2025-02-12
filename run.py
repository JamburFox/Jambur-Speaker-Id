import os
import torch

from utils import load_audio, extract_audio_features
from models import JamburSpeakerId

if __name__ == "__main__":
    INPUT_DIM = 13
    EMBEDDING_DIM = 128
    NUM_CLASSES = 40
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio, sr = load_audio(f'{os.getcwd()}/test.wav')
    audio_features = extract_audio_features(audio, sr)
    audio_features = torch.from_numpy(audio_features).to(device).unsqueeze(dim=0)

    model = JamburSpeakerId(INPUT_DIM, EMBEDDING_DIM, NUM_CLASSES).to(device)
    
    model.eval()
    with torch.inference_mode():
        print("audio features shape", audio_features.shape)
        embeddings = model.embedding(audio_features)
        print("embeddings output shape", embeddings.shape)