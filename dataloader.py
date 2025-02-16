from torch.utils.data import Dataset, DataLoader, random_split
import csv
import numpy as np
import torch
import random

from jambur_speaker_id.utils import load_audio, extract_audio_features

class AudioDataset(Dataset):
    def __init__(self, directory: str):
        self.directory = directory
        self.audio_files = []
        self.audio_labels = []

        self.label_to_indices = {}#maps speaker labels to audio file indicies
        self.unique_labels = []#each unique label the map has

        with open(f"{directory}/speakers.csv", newline='') as file:
            reader = csv.reader(file, delimiter=',')
            for i, row in enumerate(reader):
                file_name = str(row[0])
                label = int(row[1])
                self.audio_files.append(file_name)
                self.audio_labels.append(label)

                if label not in self.label_to_indices:
                    self.label_to_indices[label] = []
                self.label_to_indices[label].append(i)

        self.unique_labels = list(self.label_to_indices.keys())

    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, index: int):
        use_same_speaker = random.random() <= 0.5
        if use_same_speaker:
            label = np.random.choice(self.unique_labels)
            indices = self.label_to_indices[label]

            if len(indices) < 2:
                return self.__getitem__(index)# if there isnt enough samples for this index try again
            
            index1, index2 = np.random.choice(indices, 2, replace=False)
        else:
            label1, label2 = np.random.choice(self.unique_labels, 2, replace=False)
            index1 = np.random.choice(self.label_to_indices[label1])
            index2 = np.random.choice(self.label_to_indices[label2])

        file_path1 = f"{self.directory}/audio/{self.audio_files[index1]}"
        file_path2 = f"{self.directory}/audio/{self.audio_files[index2]}"
        audio1, sr1 = load_audio(file_path1)
        audio2, sr2 = load_audio(file_path2)
        audio_features1 = extract_audio_features(audio1, sr1)
        audio_features2 = extract_audio_features(audio2, sr2)
        label = 1 if self.audio_labels[index1] == self.audio_labels[index2] else 0

        return audio_features1, audio_features2, label
    

def pad_collate_fn(batch: list[tuple[np.ndarray, np.ndarray, str]]):
    spectrograms1, spectrograms2, labels = zip(*batch)
    batch_size = len(spectrograms1)

    max_length1 = max(spectrogram.shape[0] for spectrogram in spectrograms1)
    max_length2 = max(spectrogram.shape[0] for spectrogram in spectrograms2)

    #pad with 0
    spectrograms_padded1 = torch.zeros(batch_size, max_length1, spectrograms1[0].shape[1])#-1
    spectrograms_padded2 = torch.zeros(batch_size, max_length2, spectrograms2[0].shape[1])#-1

    for i in range(batch_size):
        spectrogram1 = spectrograms1[i]
        spectrogram2 = spectrograms2[i]
        spectrograms_padded1[i, :spectrogram1.shape[0], :] = torch.from_numpy(spectrogram1)
        spectrograms_padded2[i, :spectrogram2.shape[0], :] = torch.from_numpy(spectrogram2)

    torch_labels = torch.tensor(labels).unsqueeze(dim=1)

    return spectrograms_padded1, spectrograms_padded2, torch_labels
    
def get_data_loaders(directory: str, batch_size: int = 32, test_split: float = 0.2) -> tuple[DataLoader, DataLoader]:
    dataset = AudioDataset(directory)

    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)

    return train_loader, test_loader
