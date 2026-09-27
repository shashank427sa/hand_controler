# gesture_lstm.py
import torch
import torch.nn as nn

class TemporalGestureLSTM(nn.Module):
    def __init__(self, input_size=42, hidden_dim=64, num_layers=2, num_classes=5):
        super(TemporalGestureLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # Input shape: [batch_size, sequence_length=30, features=42]
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        logits = self.fc(last_step)
        return logits