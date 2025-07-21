import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

train_transform = transforms.Compose([transforms.ToTensor(),])
test_transform = transforms.Compose([transforms.ToTensor(),])
train_data = dsets.MNIST(root='./data', train=True, download=True, transform=train_transform)
test_data = dsets.MNIST(root='./data', train=False, download=True, transform=test_transform)
trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
testloader = DataLoader(test_data, batch_size=32, shuffle=True)