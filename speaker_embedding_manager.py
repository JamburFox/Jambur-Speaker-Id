import os
import torch
import numpy as np

from model_manager import load_speaker_id_model
from utils import load_audio_features
from models import AudioEmbedding

def get_embedding(audio_file: str, model: AudioEmbedding, device: str) -> torch.Tensor:
    model.eval()
    with torch.inference_mode():
        audio_features = load_audio_features(audio_file).to(device)
        embeddings = model(audio_features)
    return embeddings

#save under ./embeddings/speaker_id/embedding.npy
def save_new_voice_embedding(speaker_id: str, audio_file: str, save_name: str, model: AudioEmbedding, device: str):
    save_path = f"{os.getcwd()}/embeddings/{speaker_id}"

    os.makedirs(save_path, exist_ok=True)
    embeddings = get_embedding(audio_file, model, device)
    embeddings_np = embeddings.cpu().numpy()
    np.save(f"{save_path}/{save_name}.npy", embeddings_np)

def load_voice_embedding(embedding_path: str):
    embedding = torch.from_numpy(np.load(embedding_path))
    return embedding

def get_speaker_id_dirs() -> list:
    embeddings_path = f"{os.getcwd()}/embeddings"
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

def compare_embeddings(embedding_1: torch.tensor, embedding_2: torch.tensor) -> float:
    difference = torch.sum(torch.abs(embedding_1 - embedding_2))
    return difference.item()

def scan_embeddings_best_match(match_embedding: torch.tensor, log: bool=False):
    best_score = 0
    best_speaker_id = None

    dirs = get_speaker_id_dirs()
    for dir in dirs:
        if log:
            print(f"==={dir.name}===")
        files = get_speaker_id_files(dir.path)
        for file in files:
            embedding = load_voice_embedding(file.path)
            diff = compare_embeddings(match_embedding, embedding)
            if log:
                print("-", file.name, "|", diff)

            if best_speaker_id == None or diff <= best_score:
                best_score = diff
                best_speaker_id = dir.name
    if log:
        print("Best speaker match:", best_speaker_id)