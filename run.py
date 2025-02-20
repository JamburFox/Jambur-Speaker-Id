import os
import torch
import argparse
import numpy as np

from jambur_speaker_id.model_manager import load_speaker_id_model, run_model_file, DEFAULT_MODEL_PATH

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the speaker id model.')
    parser.add_argument('--audio_file', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.wav"), help='the source audio file')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH, help='location of the model')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='the device to use')
    args = parser.parse_args()

    model = load_speaker_id_model(args.model).to(args.device)
    run_model_file(model, args.audio_file, args.device, True)