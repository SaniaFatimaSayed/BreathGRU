import torch
import torch.nn as nn

class BreathGRUSegmenter(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 96, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out)
