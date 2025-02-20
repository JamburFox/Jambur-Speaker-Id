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
    def __init__(self, input_dim: int, embedding_dim: int, attention_dim: int, num_attention_heads: int, lstm_hidden_size: int, lstm_num_layers: int):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.num_attention_heads = num_attention_heads
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers

        self.cnn_network = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=16, kernel_size=(3,3), stride=1, padding=(1,1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=1, padding=(1,1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=1, padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.attention = MultiHeadAttention(64, attention_dim, num_attention_heads)
        self.attention_norm = nn.LayerNorm(64)

        self.feed_forward = PositionwiseFeedForward(64, 256)
        self.feed_forward_norm = nn.LayerNorm(64)

        self.lstm = nn.LSTM(input_size=64, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers, batch_first=True, bidirectional=True, dropout=0.2)

        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size*2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #cnn network
        x = x.permute(0, 2, 1).unsqueeze(-1)
        x = self.cnn_network(x)
        x = x.squeeze(dim=3).permute(0, 2, 1)#[batch, seq, hidden_size]

        #attention
        attended_x = self.attention(x)#[batch, seq, input_dim]
        attended_x = self.attention_norm(attended_x + x)#residual connection
        
        #position-wise feed forward
        ffn_x = self.feed_forward(attended_x)
        ffn_x = self.feed_forward_norm(ffn_x + attended_x)#Residual connection

        #LSTM
        lstm_out, _ = self.lstm(ffn_x)#[batch, seq, hidden_size*2] due to bidirectional: hidden_size * 2
        lstm_out = lstm_out[:, -1, :]#[batch, hidden_size*2] Use the last LSTM output #[1, hidden_size*2]

        embeddings = self.fc(lstm_out)
        return embeddings