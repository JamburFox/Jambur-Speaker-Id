import os
import torch
import numpy as np

from .model import JamburSpeakerId, AudioEmbedding
from .speaker_embedding_manager import get_embedding, scan_embeddings_best_match
from .utils import load_audio

INPUT_DIM = 13
EMBEDDING_DIM = 128
NUM_CLASSES = 40
SAVE_PATH = f"{os.path.dirname(os.path.abspath(__file__))}/models"
DEFAULT_MODEL_PATH = f"{SAVE_PATH}/jambur_speaker_id.pt"
DEFAULT_EMBEDDING_PATH = f"{SAVE_PATH}/jambur_speaker_id_e.pt"

def run_model_audio(model: AudioEmbedding, audio: np.ndarray, sample_rate: int, device: str, log_output: bool=False) -> str:
    embeddings = get_embedding(audio, sample_rate, model, device).cpu()
    return scan_embeddings_best_match(embeddings, log_output)

def run_model_file(model: AudioEmbedding, audio_file: str, device: str, log_output: bool=False) -> str:
    audio, sr = load_audio(audio_file)
    return run_model_audio(model, audio, sr, device, log_output)

def load_speaker_id_model(file_path: str = DEFAULT_MODEL_PATH) -> JamburSpeakerId:
    model = JamburSpeakerId(INPUT_DIM, EMBEDDING_DIM, NUM_CLASSES)
    if file_path is not None:
        try:
            model.load_state_dict(torch.load(f=file_path))
        except:
            print(f"Unable to load model from {file_path}!")
    return model

def load_embedding_model(file_path: str = DEFAULT_EMBEDDING_PATH) -> AudioEmbedding:
    model = AudioEmbedding(INPUT_DIM, EMBEDDING_DIM)
    if file_path is not None:
        try:
            model.load_state_dict(torch.load(f=file_path))
        except:
            print(f"Unable to load model from {file_path}!")
    return model

def save_speaker_id_model(model: JamburSpeakerId, save_path: str = DEFAULT_MODEL_PATH):
    try:
        os.makedirs(SAVE_PATH, exist_ok=True)
        torch.save(obj=model.state_dict(), f=save_path)
    except:
        print(f"Unable to save model to {save_path}!")

def save_embedding_model(model: JamburSpeakerId, save_path: str = DEFAULT_EMBEDDING_PATH):
    try:
        os.makedirs(SAVE_PATH, exist_ok=True)
        torch.save(obj=model.embedding.state_dict(), f=save_path)
    except:
        print(f"Unable to save model to {save_path}!")
