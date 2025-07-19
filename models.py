import torch
from torch import nn
import logging

logger = logging.getLogger(__name__)

class KeypressGRU(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward_once(self, x):
        _, h = self.gru(x)
        return h.squeeze(0)

    def forward(self, x1, x2):
        o1, o2 = self.forward_once(x1), self.forward_once(x2)
        diff = torch.abs(o1 - o2)
        return self.fc(diff)

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

model: KeypressGRU = None
device: torch.device = None

def load_model():
    global model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KeypressGRU(input_dim=3, hidden_dim=64).to(device)
    model.load_state_dict(torch.load("siamese_gru_model.pth", map_location=device))
    model.eval()
    logger.info(f"Model loaded on {device}")



