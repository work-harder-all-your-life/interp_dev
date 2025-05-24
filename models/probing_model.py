import torch.nn as nn


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
