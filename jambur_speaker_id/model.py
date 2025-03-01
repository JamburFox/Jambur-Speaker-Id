import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim: int, attention_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.attention_dim = attention_dim

        self.query_layer = nn.Linear(input_dim, num_heads * attention_dim)
        self.key_layer = nn.Linear(input_dim, num_heads * attention_dim)
        self.value_layer = nn.Linear(input_dim, num_heads * attention_dim)

        self.fc_out = nn.Linear(num_heads * attention_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, _ = x.size()

        queries = self.query_layer(x).view(batch_size, seq_length, self.num_heads, self.attention_dim)
        keys = self.key_layer(x).view(batch_size, seq_length, self.num_heads, self.attention_dim)
        values = self.value_layer(x).view(batch_size, seq_length, self.num_heads, self.attention_dim)

        queries = queries.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 3, 1)
        values = values.permute(0, 2, 1, 3)

        #Compute attention scores using the dot product between queries and keys
        attention_scores = torch.matmul(queries, keys) / (self.attention_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)#Apply softmax to obtain attention weights
        attended_values = torch.matmul(attention_weights, values)

        attended_values = attended_values.permute(0, 2, 1, 3).contiguous()
        attended_values = attended_values.view(batch_size, seq_length, -1)#Concatenate heads

        output = self.fc_out(attended_values)#Compute the weighted sum of values
        return output

class PositionwiseFeedForward(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))

class JamburSpeakerId(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, attention_dim: int, num_attention_heads: int, initial_cnn_size: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.num_attention_heads = num_attention_heads
        self.initial_cnn_size = initial_cnn_size
        self.hidden_dim = hidden_dim

        self.cnn_network = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=initial_cnn_size, kernel_size=(3,3), stride=1, padding=(1,1)),
            nn.BatchNorm2d(initial_cnn_size),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),#[bins,seq/2]

            nn.Conv2d(in_channels=initial_cnn_size, out_channels=initial_cnn_size*2, kernel_size=(3,3), stride=1, padding=(1,1)),
            nn.BatchNorm2d(initial_cnn_size*2),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),#[bins,seq/4]

            nn.Conv2d(in_channels=initial_cnn_size*2, out_channels=initial_cnn_size*4, kernel_size=(3,3), stride=1, padding=(1,1)),
            nn.BatchNorm2d(initial_cnn_size*4),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),#[bins/2,seq/8]
        )

        self.cnn_dense = nn.Sequential(
            nn.Linear(input_dim*initial_cnn_size*4, self.hidden_dim),
            nn.ReLU(),
        )

        self.attention = MultiHeadAttention(self.hidden_dim, attention_dim, num_attention_heads)
        self.attention_norm = nn.LayerNorm(self.hidden_dim)

        self.feed_forward = PositionwiseFeedForward(self.hidden_dim, self.hidden_dim*4)
        self.feed_forward_norm = nn.LayerNorm(self.hidden_dim)

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #cnn network
        x = x.permute(0, 2, 1).unsqueeze(1)
        x = self.cnn_network(x)
        x = x.permute(0, 3, 1, 2).flatten(-2)#[batch, seq, channels*mels]

        #cnn dense
        x = self.cnn_dense(x)#[batch, seq, hidden_size]

        #attention
        attended_x = self.attention(x)#[batch, seq, hidden_size]
        attended_x = self.attention_norm(attended_x + x)#residual connection
        
        #position-wise feed forward
        ffn_x = self.feed_forward(attended_x)
        ffn_x = self.feed_forward_norm(ffn_x + attended_x)#Residual connection #[batch, seq, dim]

        #global average pooling across the sequence dimension
        pooled_x = torch.mean(ffn_x, dim=1)  # [batch, 64]

        embeddings = self.fc(pooled_x)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings