import os
import numpy as np
import torch
import torch.nn as nn
import argparse

from jambur_speaker_id.speaker_embedding_manager import save_new_voice_embedding
from jambur_speaker_id.model_manager import load_speaker_id_model, DEFAULT_MODEL_PATH

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a voice embedding from an audio file.')
    parser.add_argument('--speaker_id', type=str, default="jambur", help='the speaker id')
    parser.add_argument('--audio_file', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.wav"), help='the source audio file')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH, help='location of the model')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='the device to use')
    args = parser.parse_args()

    model = load_speaker_id_model(args.model).to(args.device)

    if os.path.isfile(args.audio_file):
        save_name = os.path.splitext(os.path.basename(args.audio_file))[0]
        save_new_voice_embedding(args.speaker_id, args.audio_file, save_name, model, args.device)
    elif os.path.isdir(args.audio_file):
        with os.scandir(args.audio_file) as entries:
            for entry in entries:
                if entry.is_file():
                    try:
                        save_name = entry.name.split(".")[0]
                        save_new_voice_embedding(args.speaker_id, entry.path, save_name, model, args.device)
                    except:
                        print(f"Failed to load {entry.name}!")


