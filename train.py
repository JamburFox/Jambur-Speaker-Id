import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import argparse

from jambur_speaker_id.model import SpeakerIdEmbedding
from jambur_speaker_id.model_manager import load_speaker_id_model, save_speaker_id_model
from dataloader import get_data_loaders

def train_step(data_loader: torch.utils.data.DataLoader, model: SpeakerIdEmbedding, optimizer: torch.optim.Optimizer, device: str):
    running_loss = 0

    model.train()
    for batch, (audio1, audio2, labels) in enumerate(data_loader):
        audio1 = audio1.to(device)
        audio2 = audio2.to(device)
        labels = labels.to(device).squeeze()

        embeddings1 = model(audio1)
        embeddings2 = model(audio2)

        loss = contrastive_loss(embeddings1, embeddings2, labels)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1 == 0:
            print(f"=== Train Batch {batch+1} / {len(data_loader)} ===", end="\r")

    avg_loss = running_loss / len(data_loader)
    return avg_loss

def test_step(data_loader: torch.utils.data.DataLoader, model: SpeakerIdEmbedding, device: str):
    running_loss = 0

    model.eval()
    with torch.no_grad():
        for batch, (audio1, audio2, labels) in enumerate(data_loader):
            audio1 = audio1.to(device)
            audio2 = audio2.to(device)
            labels = labels.to(device).squeeze()

            embeddings1 = model(audio1)
            embeddings2 = model(audio2)

            loss = contrastive_loss(embeddings1, embeddings2, labels)
            running_loss += loss.item()

            if batch % 1 == 0:
                print(f"=== Test Batch {batch+1} / {len(data_loader)} ===", end="\r")

    avg_loss = running_loss / len(data_loader)
    return avg_loss

def contrastive_loss(output1, output2, label, margin=1.0):
    euclidean_distance = F.pairwise_distance(output1, output2)
    loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                  (label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
    return loss_contrastive

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the speaker id model.')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--scheduler_step', type=int, default=10, help='learning rate')
    parser.add_argument('--scheduler_gamma', type=float, default=0.5, help='learning rate')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='the device to use')
    args = parser.parse_args()

    best_validation_loss = float('inf')

    model = load_speaker_id_model().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)

    train_loader, test_loader = get_data_loaders('./dataset', args.batch_size, 0.2)
    print(f"Train Size: {len(train_loader)} | test Size: {len(test_loader)} | Scheduler Step: {args.scheduler_step}")

    for epoch in range(0, args.epochs):
        train_loss = train_step(train_loader, model, optimizer, args.device)
        test_loss = test_step(test_loader, model, args.device)

        if test_loss <= best_validation_loss:
            best_validation_loss = test_loss
            save_speaker_id_model(model)
            print(f"Saving new best Model! | Best Loss: {best_validation_loss:.4f}!\n")

        scheduler.step()
        print(f"=== Epoch: {epoch+1} / {args.epochs} | Train_loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Learning Rate: {scheduler.get_last_lr()[0]:.6f}===")

