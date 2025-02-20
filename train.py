import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import argparse
import os
from jambur_speaker_id.model import JamburSpeakerId
from jambur_speaker_id.model_manager import load_speaker_id_model, save_speaker_id_model, get_save_path
from dataloader import get_data_loaders

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters())

def compare_embeddings_cosine(embeddings1: torch.Tensor, embeddings2: torch.Tensor):
    query_norm = F.normalize(embeddings1, dim=1)
    reference_norms = F.normalize(embeddings2, dim=1)
    similarities = (query_norm * reference_norms).sum(dim=1)#Element-wise multiply and sum along dimensions
    return similarities

def test_accuracy(embeddings1: torch.Tensor, embeddings2: torch.Tensor, labels: torch.Tensor, threshold: float = 0.85):
    similarities = compare_embeddings_cosine(embeddings1, embeddings2)

    predicted_labels = (similarities >= threshold).int()#for each item 1 if greater than threshold else 0
    correct_predictions = (predicted_labels == labels)#true if predicted and actual labels are the same otherwise false
    accuracy = correct_predictions.sum().item() / len(labels)#get accuracy of correct labels between 0 and 1
    return accuracy

def test_accuracy_triplet(anchor_embeddings: torch.Tensor, positive_embeddings: torch.Tensor, negative_embeddings: torch.Tensor, threshold: float = 0.8):
    accuracy_positive = test_accuracy(anchor_embeddings, positive_embeddings, torch.ones(anchor_embeddings.size(0), dtype=torch.int, device=anchor_embeddings.device), threshold)
    accuracy_negative = test_accuracy(anchor_embeddings, negative_embeddings, torch.zeros(anchor_embeddings.size(0), dtype=torch.int, device=anchor_embeddings.device), threshold)
    accuracy = (accuracy_positive + accuracy_negative) / 2.0
    return accuracy

def train_step(data_loader: torch.utils.data.DataLoader, model: JamburSpeakerId, optimizer: torch.optim.Optimizer, device: str):
    running_loss = 0
    running_accuracy = 0

    model.train()
    for batch, (anchor, positive, negative) in enumerate(data_loader):
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        anchor_embeddings = model(anchor)
        positive_embeddings = model(positive)
        negative_embeddings = model(negative)

        loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
        running_loss += loss.item()

        batch_accuracy = test_accuracy_triplet(anchor_embeddings, positive_embeddings, negative_embeddings)
        running_accuracy += batch_accuracy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1 == 0:
            print("\033[K", end="\r")
            print(f"=== Train Batch {batch+1} / {len(data_loader)} ===", end="\r")

    avg_loss = running_loss / len(data_loader)
    avg_acc = running_accuracy / len(data_loader)
    return avg_loss, avg_acc

def test_step(data_loader: torch.utils.data.DataLoader, model: JamburSpeakerId, device: str):
    running_loss = 0
    running_accuracy = 0

    model.eval()
    with torch.no_grad():
        for batch, (anchor, positive, negative) in enumerate(data_loader):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            anchor_embeddings = model(anchor)
            positive_embeddings = model(positive)
            negative_embeddings = model(negative)

            loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
            running_loss += loss.item()

            batch_accuracy = test_accuracy_triplet(anchor_embeddings, positive_embeddings, negative_embeddings)
            running_accuracy += batch_accuracy

            if batch % 1 == 0:
                print("\033[K", end="\r")
                print(f"=== Test Batch {batch+1} / {len(data_loader)} ===", end="\r")

    avg_loss = running_loss / len(data_loader)
    avg_acc = running_accuracy / len(data_loader)
    return avg_loss, avg_acc

def triplet_loss(anchor, positive, negative, margin=1.0):
    positive_distance = F.pairwise_distance(anchor, positive)
    negative_distance = F.pairwise_distance(anchor, negative)
    loss_triplet = torch.mean(F.relu(positive_distance - negative_distance + margin))
    return loss_triplet

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the speaker id model.')
    parser.add_argument('--dataset', type=str, default=os.path.join(".", "dataset"), help='location of the dataset')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--scheduler_step', type=int, default=5, help='learning rate')
    parser.add_argument('--scheduler_gamma', type=float, default=0.9, help='learning rate')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='the device to use')
    args = parser.parse_args()

    best_validation_loss = float('inf')

    model = load_speaker_id_model(get_save_path("jambur_speaker_id_latest")).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)

    print(f"Total Parameters: {count_parameters(model):,}")

    train_loader, test_loader = get_data_loaders(args.dataset, args.batch_size, 0.2)
    print(f"Train Size: {len(train_loader)} | test Size: {len(test_loader)} | Scheduler Step: {args.scheduler_step}")

    torch.cuda.empty_cache()

    for epoch in range(0, args.epochs):
        train_loss, train_acc = train_step(train_loader, model, optimizer, args.device)
        test_loss, test_acc = test_step(test_loader, model, args.device)
        print(f"=== Epoch: {epoch+1} / {args.epochs} | Train_loss: {train_loss:.4f} | Train Accuracy: {train_acc*100:.2f}%   | Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc*100:.2f}%  | Learning Rate: {scheduler.get_last_lr()[0]:.6f}===")
        
        if test_loss <= best_validation_loss:
            best_validation_loss = test_loss
            save_speaker_id_model(model)
            print(f"Saving new best Model! | Best Loss: {best_validation_loss:.4f}!")
        save_speaker_id_model(model, get_save_path("jambur_speaker_id_latest"))

        scheduler.step()

