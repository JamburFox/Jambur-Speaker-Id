from torch.utils.data import Dataset, DataLoader, random_split, Subset
import csv
import numpy as np
import torch
import random

from utils import load_audio, extract_audio_features

class AudioDataset(Dataset):
    def __init__(self, directory: str):
        self.directory = directory
        self.audio_files = []
        self.audio_labels = []

        with open(f"{directory}/speakers.csv", newline='') as file:
            reader = csv.reader(file, delimiter=',')
            for row in reader:
                file_name = str(row[0])
                label = int(row[1])
                self.audio_files.append(file_name)
                self.audio_labels.append(label)

    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, index: int):
        file_path = f"{self.directory}/audio/{self.audio_files[index]}"
        audio, sr = load_audio(file_path)
        audio_features = extract_audio_features(audio, sr)
        label = self.audio_labels[index]

        return audio_features, label
    

def pad_collate_fn(batch: list[tuple[np.ndarray, str]]):
    spectrograms, labels = zip(*batch)
    batch_size = len(spectrograms)

    max_length = max(spectrogram.shape[0] for spectrogram in spectrograms)
    spectrograms_padded = torch.zeros(batch_size, max_length, spectrograms[0].shape[1])-1#pad with -1
    spectrograms_mask = torch.zeros(batch_size, max_length).float()#mask where 1 = valid data and 0 = padding
    for i, spectrogram in enumerate(spectrograms):
        spectrograms_padded[i, :spectrogram.shape[0], :] = torch.from_numpy(spectrogram)
        spectrograms_mask[i, :spectrogram.shape[0]] = torch.ones(spectrogram.shape[0])

    torch_labels = torch.tensor(labels).unsqueeze(dim=1)

    return spectrograms_padded, torch_labels#can return mask if needed in the future
    
def get_data_loaders(directory: str, batch_size: int = 32, test_split: float = 0.2):
    dataset = AudioDataset(directory)

    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)

    return train_loader, test_loader