import os
import torch
import numpy as np
import json

from .model import JamburSpeakerId
from .speaker_embedding_manager import get_embedding, scan_embeddings_best_match
from .utils import load_audio

#, lstm_hidden_size: int = 256, num_lstm_layers: int = 2
INPUT_DIM = 13
EMBEDDING_DIM = 256
ATTENTION_DIM = 16
NUM_ATTENTION_HEADS = 4
LSTM_HIDDEN_SIZE = 32
LSTM_NUM_LAYERS = 2
SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEFAULT_MODEL_PATH = os.path.join(SAVE_PATH, "jambur_speaker_id.pt")

def run_model_audio(model: JamburSpeakerId, audio: np.ndarray, sample_rate: int, device: str, log_output: bool=False) -> str:
    embeddings = get_embedding(audio, sample_rate, model, device).cpu()
    return scan_embeddings_best_match(embeddings, log_output)

def run_model_file(model: JamburSpeakerId, audio_file: str, device: str, log_output: bool=False) -> str:
    audio, sr = load_audio(audio_file)
    return run_model_audio(model, audio, sr, device, log_output)

def load_speaker_id_model(file_path: str = DEFAULT_MODEL_PATH) -> JamburSpeakerId:
    model = JamburSpeakerId(INPUT_DIM, EMBEDDING_DIM, ATTENTION_DIM, NUM_ATTENTION_HEADS, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS)
    if file_path is not None:
        try:
            model.load_state_dict(torch.load(f=file_path))
        except:
            print(f"Unable to load model from {file_path}!")
    return model

def get_save_path(save_name: str):
    return os.path.join(SAVE_PATH, f"{save_name}.pt")

def save_speaker_id_model(model: JamburSpeakerId, save_path: str = DEFAULT_MODEL_PATH):
    try:
        os.makedirs(SAVE_PATH, exist_ok=True)
        torch.save(obj=model.state_dict(), f=save_path)

        model_metadata = {
            "input_dim": INPUT_DIM,
            "embedding_dim": EMBEDDING_DIM,
            "attention_dim": ATTENTION_DIM,
            "num_attention_heads": NUM_ATTENTION_HEADS,
            "lstm_hidden_size": LSTM_HIDDEN_SIZE,
            "lstm_num_layers": LSTM_NUM_LAYERS
        }
        (file_name, _) = os.path.splitext(os.path.basename(save_path))
        with open(os.path.join(os.path.dirname(save_path), f'{file_name}.json'), 'w') as file:
            json.dump(model_metadata, file, indent=4)
    except Exception as e:
        print(f"Unable to save model to {save_path}! {e}")