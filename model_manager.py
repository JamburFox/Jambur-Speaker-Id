import os
import torch

from models import JamburSpeakerId, AudioEmbedding

INPUT_DIM = 13
EMBEDDING_DIM = 128
NUM_CLASSES = 40
DEFAULT_MODEL_PATH = f"{os.getcwd()}/models/jambur_speaker_id.py"
DEFAULT_EMBEDDING_PATH = f"{os.getcwd()}/models/jambur_speaker_id_e.py"

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
        torch.save(obj=model.state_dict(), f=save_path)
    except:
        print(f"Unable to save model to {save_path}!")

def save_embedding_model(model: JamburSpeakerId, save_path: str = DEFAULT_EMBEDDING_PATH):
    try:
        torch.save(obj=model.embedding.state_dict(), f=save_path)
    except:
        print(f"Unable to save model to {save_path}!")