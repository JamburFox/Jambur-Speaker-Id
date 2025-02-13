import os
import torch
import argparse

from speaker_embedding_manager import get_embedding, scan_embeddings_best_match
from model_manager import load_embedding_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a voice embedding from an audio file.')
    parser.add_argument('--audio_file', type=str, default=f'{os.path.dirname(os.path.abspath(__file__))}/test.wav', help='the source audio file')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='the device to use')
    args = parser.parse_args()

    model = load_embedding_model().to(args.device)
    match_embeddings = get_embedding(args.audio_file, model, args.device).cpu()
    scan_embeddings_best_match(match_embeddings, True)