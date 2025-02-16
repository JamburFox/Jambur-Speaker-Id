import torch
import torch.nn as nn

class SpeakerIdEmbedding(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int):
        super().__init__()

        self.conv_embedding = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=32, kernel_size=(3,3), stride=1, padding=(1,1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=1, padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=1, padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )
        #self.fc = nn.Linear(128, embedding_dim)
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1).unsqueeze(-1)
        x = self.conv_embedding(x)
        embeddings = self.fc(x)
        return embeddings