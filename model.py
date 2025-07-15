import torch
import torch.nn as nn
import torch.nn.functional as F

class SensorNetwork(nn.Module):
    def __init__(self, input_size, embedding_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward_once(self, x):
        return self.fc(x)

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)