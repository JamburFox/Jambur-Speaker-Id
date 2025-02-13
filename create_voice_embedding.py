import os
import numpy as np
import torch
import torch.nn as nn
import argparse

from speaker_embedding_manager import save_new_voice_embedding
from model_manager import load_embedding_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a voice embedding from an audio file.')
    parser.add_argument('--speaker_id', type=str, default="jambur", help='the speaker id')
    parser.add_argument('--audio_file', type=str, default=f'{os.path.dirname(os.path.abspath(__file__))}/test.wav', help='the source audio file')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='the device to use')
    args = parser.parse_args()

    save_name = os.path.splitext(os.path.basename(args.audio_file))[0]

    model = load_embedding_model().to(args.device)
    save_new_voice_embedding(args.speaker_id, args.audio_file, save_name, model, args.device)



