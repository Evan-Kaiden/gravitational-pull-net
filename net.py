import torch.nn as nn
import torch.nn.functional as F

class network(nn.Module):
    def __init__(self, n_inputs, classes=10):
        super().__init__()

        self.fc1 = nn.Linear(n_inputs, classes)

    def forward(self, x):
        x = x.flatten(1)
        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)
        return x