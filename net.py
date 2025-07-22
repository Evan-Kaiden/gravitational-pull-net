import torch.nn as nn
import torch.nn.functional as F

class network(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)  
        self.pool = nn.MaxPool2d(2, 2)                                      
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)           
        
        self.fc1 = nn.Linear(32 * 8 * 8, num_classes)  

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)
