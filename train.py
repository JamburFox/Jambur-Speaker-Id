import torch
import torch.nn as nn
import argparse

from models import JamburSpeakerId
from model_manager import load_speaker_id_model, save_speaker_id_model, save_embedding_model
from models import JamburSpeakerId
from dataloader import get_data_loaders
from utils import accuracy


def train_step(data_loader: torch.utils.data.DataLoader, model: JamburSpeakerId, criterion: nn.CrossEntropyLoss, optimizer: torch.optim.Optimizer, device: str):
    running_loss = 0

    model.train()
    for batch, (audio, labels) in enumerate(data_loader):
        audio = audio.to(device)
        labels = labels.to(device).squeeze()

        embeddings, output = model(audio)

        loss = criterion(output, labels)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1 == 0:
            print(f"=== Train Batch {batch+1} / {len(data_loader)} ===", end="\r")

    avg_loss = running_loss / len(data_loader)
    return avg_loss

def test_step(data_loader: torch.utils.data.DataLoader, model: JamburSpeakerId, criterion: nn.CrossEntropyLoss, device: str):
    running_loss = 0
    acc = 0

    model.eval()
    with torch.no_grad():
        for batch, (audio, labels) in enumerate(data_loader):
            audio = audio.to(device)
            labels = labels.to(device).squeeze()

            embeddings, output = model(audio)
            probabilities = torch.softmax(output, dim=1)
            predicted_labels = torch.argmax(probabilities, dim=1)

            #loss = criterion(predicted_labels, labels)
            loss = criterion(output, labels)
            acc += accuracy(labels, predicted_labels)

            running_loss += loss.item()

            if batch % 1 == 0:
                print(f"=== Test Batch {batch+1} / {len(data_loader)} ===", end="\r")

    avg_accuracy = acc / len(data_loader)
    avg_loss = running_loss / len(data_loader)
    return avg_loss, avg_accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the speaker id model.')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='the device to use')
    args = parser.parse_args()

    model = load_speaker_id_model().to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    train_loader, test_loader = get_data_loaders('./dataset', args.batch_size, 0.2)
    print(f"Train Size: {len(train_loader)} | test Size: {len(test_loader)}")

    for epoch in range(0, args.epochs):
        train_loss = train_step(train_loader, model, criterion, optimizer, args.device)
        test_loss, test_accuracy = test_step(test_loader, model, criterion, args.device)
        print(f"=== Epoch: {epoch+1} / {args.epochs} | Train_loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}% ===")
        save_speaker_id_model(model)
        save_embedding_model(model)

