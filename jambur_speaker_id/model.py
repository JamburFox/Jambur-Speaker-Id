import torch
import torch.nn as nn

class AudioEmbedding(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int):
        super().__init__()

        self.conv_embedding = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=32, kernel_size=(3,3), stride=1, padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=1, padding=(1,1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )
        self.fc = nn.Linear(64, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1).unsqueeze(-1)
        x = self.conv_embedding(x)
        embeddings = self.fc(x)
        return embeddings
    
class JamburSpeakerId(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, num_classes: int):
        super().__init__()

        self.embedding = AudioEmbedding(input_dim, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.embedding(x)
        output = self.classifier(embeddings)

        return embeddings, output