import os
import torch
import numpy as np
import torch.nn.functional as F

from .utils import load_audio_features, extract_audio_features, load_audio
from .model import SpeakerIdEmbedding

def get_embedding(audio: np.ndarray, sample_rate: int, model: SpeakerIdEmbedding, device: str) -> torch.Tensor:
    model.eval()
    with torch.inference_mode():
        audio_features = extract_audio_features(audio, sample_rate)
        audio_features = torch.from_numpy(audio_features).unsqueeze(dim=0).to(device)
        embeddings = model(audio_features)
    return embeddings

#save under ./embeddings/speaker_id/embedding.npy
def save_new_voice_embedding(speaker_id: str, audio_file: str, save_name: str, model: SpeakerIdEmbedding, device: str):
    save_path = f"{os.path.dirname(os.path.abspath(__file__))}/embeddings/{speaker_id}"

    os.makedirs(save_path, exist_ok=True)
    audio, sr = load_audio(audio_file)
    embeddings = get_embedding(audio, sr, model, device)
    embeddings_np = embeddings.cpu().numpy()
    np.save(f"{save_path}/{save_name}.npy", embeddings_np)

def load_voice_embedding(embedding_path: str):
    embedding = torch.from_numpy(np.load(embedding_path))
    return embedding

def get_speaker_id_dirs() -> list:
    embeddings_path = f"{os.path.dirname(os.path.abspath(__file__))}/embeddings"
    dirs = []
    try:
        with os.scandir(embeddings_path) as entries:
            for dir in entries:
                if dir.is_dir():
                    dirs.append(dir)
    except:
        print("Unable to load embeddings path")
    return dirs

def get_speaker_id_files(speaker_id_path: str) -> list:
    embedding_files = []
    try:
        with os.scandir(speaker_id_path) as files:
            for file in files:
                if file.is_file():
                    embedding_files.append(file)
    except:
        print("Unable to load speaker embedding files")
    return embedding_files

def compare_embeddings(embedding_1: torch.Tensor, embedding_2: torch.Tensor) -> float:
    difference = torch.sum(torch.abs(embedding_1 - embedding_2))
    return difference.item()

def compare_embeddings_cosine(embedding_1: torch.Tensor, embedding_2: torch.Tensor) -> float:
    query_norm = F.normalize(embedding_1, dim=1)
    reference_norms = F.normalize(embedding_2, dim=1)

    similarities = torch.mm(reference_norms, query_norm.t()).squeeze()
    return similarities

def scan_embeddings_best_match(match_embedding: torch.Tensor, log_output: bool=False) -> str:
    best_score = 0
    best_speaker_id = None

    dirs = get_speaker_id_dirs()
    for dir in dirs:
        if log_output:
            print(f"==={dir.name}===")
        files = get_speaker_id_files(dir.path)
        for file in files:
            embedding = load_voice_embedding(file.path)
            diff = compare_embeddings_cosine(match_embedding, embedding)
            if log_output:
                print("-", file.name, "|", diff)

            if best_speaker_id == None or diff > best_score:#<= if using compare_embeddings
                best_score = diff
                best_speaker_id = dir.name
    if log_output:
        print("Best speaker match:", best_speaker_id)
    return best_speaker_id