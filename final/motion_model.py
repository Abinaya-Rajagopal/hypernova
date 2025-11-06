import torch
import torch.nn as nn
import numpy as np

class MotionModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=16, output_size=3):
        super(MotionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # take the last time step
        return out

    def predict(self, sensor_window):
        """
        Wrapper function to take a list of sensor readings (like [[0.1, 0.5, 0.2], [0.15, 0.55, 0.25]])
        and return numpy output for the controller.
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor(sensor_window, dtype=torch.float32).unsqueeze(0)  # shape (1, seq, features)
            y = self.forward(x)
            return y.squeeze(0).numpy()
