import torch.nn as nn


class EmbeddingCls(nn.Module):
    """
    Baseline model class for embeddings gender classification
    """

    def __init__(self, input_dim=256, num_classes=2):
        super(EmbeddingCls, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x1)
        return x1, x2


class ProbingCls(nn.Module):
    """
    Baseline model class for probing
    """

    def __init__(self, input_size, num_classes=1):
        super(ProbingCls, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.act(self.fc(x))
