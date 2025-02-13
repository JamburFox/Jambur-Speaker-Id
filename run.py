import os
import torch
import argparse
import numpy as np

from model_manager import load_embedding_model, run_model_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the speaker id model.')
    parser.add_argument('--audio_file', type=str, default=f'{os.path.dirname(os.path.abspath(__file__))}/test.wav', help='the source audio file')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='the device to use')
    args = parser.parse_args()

    model = load_embedding_model().to(args.device)
    run_model_file(model, args.audio_file, args.device)