import torch.nn as nn


class EmbeddingModel(nn.Module):
    """
    Baseline model class for embeddings
    """

    def __init__(self, input_dim=256, hidden_dim=128, num_classes=2):
        super(EmbeddingModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x1)
        return x1, x2
