import os
import torch
import numpy as np
import json
import time

from .model import JamburSpeakerId
from .speaker_embedding_manager import get_embedding, scan_embeddings_best_match
from .utils import load_audio

DEFAULT_INPUT_DIM = 13
DEFAULT_EMBEDDING_DIM = 256
DEFAULT_ATTENTION_DIM = 16
DEFAULT_NUM_ATTENTION_HEADS = 4
DEFAULT_LSTM_HIDDEN_SIZE = 32
DEFAULT_LSTM_NUM_LAYERS = 2
SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEFAULT_MODEL_PATH = os.path.join(SAVE_PATH, "jambur_speaker_id.pt")

def run_model_audio(model: JamburSpeakerId, audio: np.ndarray, sample_rate: int, device: str, log_output: bool=False) -> str:
    start_time = time.time()
    embeddings = get_embedding(audio, sample_rate, model, device).cpu()
    speaker_id = scan_embeddings_best_match(embeddings, log_output)
    duration = time.time() - start_time

    if log_output:
        print(f"Best speaker match: {speaker_id} | Execution Time: {duration:.4}s")
    return speaker_id

def run_model_file(model: JamburSpeakerId, audio_file: str, device: str, log_output: bool=False) -> str:
    audio, sr = load_audio(audio_file)
    return run_model_audio(model, audio, sr, device, log_output)

def load_speaker_id_model(file_path: str = DEFAULT_MODEL_PATH) -> JamburSpeakerId:
    try:
        (file_name, _) = os.path.splitext(os.path.basename(file_path))
        with open(os.path.join(os.path.dirname(file_path), f'{file_name}.json'), 'r') as file:
            data = json.load(file)
    except:
        print("Unable to load .json from: Instantiating with defualt values!")
        data = {}

    input_dim = data.get('input_dim', DEFAULT_INPUT_DIM)
    embedding_dim = data.get('embedding_dim', DEFAULT_EMBEDDING_DIM)
    attention_dim = data.get('attention_dim', DEFAULT_ATTENTION_DIM)
    num_attention_heads = data.get('num_attention_heads', DEFAULT_NUM_ATTENTION_HEADS)
    lstm_hidden_size = data.get('lstm_hidden_size', DEFAULT_LSTM_HIDDEN_SIZE)
    lstm_num_layers = data.get('lstm_num_layers', DEFAULT_LSTM_NUM_LAYERS)

    model = JamburSpeakerId(input_dim, embedding_dim, attention_dim, num_attention_heads, lstm_hidden_size, lstm_num_layers)
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
            "input_dim": model.input_dim,
            "embedding_dim": model.embedding_dim,
            "attention_dim": model.attention_dim,
            "num_attention_heads": model.num_attention_heads,
            "lstm_hidden_size": model.lstm_hidden_size,
            "lstm_num_layers": model.lstm_num_layers
        }
        (file_name, _) = os.path.splitext(os.path.basename(save_path))
        with open(os.path.join(os.path.dirname(save_path), f'{file_name}.json'), 'w') as file:
            json.dump(model_metadata, file, indent=4)
    except Exception as e:
        print(f"Unable to save model to {save_path}! {e}")