from torch.utils.data import Dataset, DataLoader, random_split
import csv
import numpy as np
import torch
import os

from jambur_speaker_id.utils import load_audio, extract_audio_features

class AudioDataset(Dataset):
    def __init__(self, directory: str, speakers_file_name: str):
        self.directory = directory
        self.audio_files = []
        self.audio_labels = []

        self.label_to_indices = {}
        self.unique_labels = []

        print("Loading Dataset...", end="\r")
        with open(os.path.join(directory, f"{speakers_file_name}.csv"), newline='') as file:
            reader = csv.reader(file, delimiter=',')
            for i, row in enumerate(reader):
                print(f"Loading Dataset... ({i})", end="\r")
                file_name = str(row[0])
                label = int(row[1])

                self.audio_files.append(os.path.join(self.directory, "audio", file_name))
                self.audio_labels.append(label)

                if label not in self.label_to_indices:
                    self.label_to_indices[label] = []
                self.label_to_indices[label].append(i)
        print("")

        #remove labels with less than 2 elements
        self.label_to_indices = {key: val for key, val in self.label_to_indices.items() if len(val) >= 2}
        self.unique_labels = np.array(list(self.label_to_indices.keys()))#make np array of unique labels

        print("Done Loading!")


    def __len__(self):
        return len(self.audio_labels)
    
    def __getitem__(self, index: int):
        label = self.audio_labels[index]
        negative_label = self.select_random_label_excluding_current_np(self.unique_labels, label)

        anchor_index, positive_index = np.random.choice(self.label_to_indices[label], 2, replace=False)
        negative_index = np.random.choice(self.label_to_indices[negative_label])

        anchor_audio, anchor_sr = load_audio(self.audio_files[anchor_index])
        positive_audio, positive_sr = load_audio(self.audio_files[positive_index])
        negative_audio, negative_sr = load_audio(self.audio_files[negative_index])
        anchor_features = extract_audio_features(anchor_audio, anchor_sr)
        positive_features = extract_audio_features(positive_audio, positive_sr)
        negative_features = extract_audio_features(negative_audio, negative_sr)

        return anchor_features, positive_features, negative_features
    
    def select_random_label_excluding_current_np(self, image_labels_array: np.ndarray, current_label: int) -> int:
        #Create a boolean mask to exclude the current label
        mask = image_labels_array != current_label

        #Use np.random.choice to select a random label from the masked array
        random_label = np.random.choice(image_labels_array[mask])
        return random_label
    
def pad_collate_fn(batch: list[tuple[np.ndarray, np.ndarray, np.ndarray]]):
    anchor_spectrograms, positive_spectrograms, negative_spectrograms = zip(*batch)
    batch_size = len(anchor_spectrograms)

    anchor_max_length = max(spectrogram.shape[0] for spectrogram in anchor_spectrograms)
    positive_max_length = max(spectrogram.shape[0] for spectrogram in positive_spectrograms)
    negative_max_length = max(spectrogram.shape[0] for spectrogram in negative_spectrograms)

    #pad with -1
    anchor_spectrograms_padded = torch.zeros(batch_size, anchor_max_length, anchor_spectrograms[0].shape[1])-1
    positive_spectrograms_padded = torch.zeros(batch_size, positive_max_length, positive_spectrograms[0].shape[1])-1
    negative_spectrograms_padded = torch.zeros(batch_size, negative_max_length, negative_spectrograms[0].shape[1])-1

    for i in range(batch_size):
        anchor_spectrogram = anchor_spectrograms[i]
        positive_spectrogram = positive_spectrograms[i]
        negative_spectrogram = negative_spectrograms[i]

        anchor_spectrograms_padded[i, :anchor_spectrogram.shape[0], :] = torch.from_numpy(anchor_spectrogram)
        positive_spectrograms_padded[i, :positive_spectrogram.shape[0], :] = torch.from_numpy(positive_spectrogram)
        negative_spectrograms_padded[i, :negative_spectrogram.shape[0], :] = torch.from_numpy(negative_spectrogram)

    return anchor_spectrograms_padded, positive_spectrograms_padded, negative_spectrograms_padded
    
def get_data_loaders(directory: str, batch_size: int = 32) -> tuple[DataLoader, DataLoader]:
    train_dataset = AudioDataset(directory, "train")
    test_dataset = AudioDataset(directory, "test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)

    return train_loader, test_loader