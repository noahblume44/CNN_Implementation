import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self):
        super.__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.final_layer = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x)
        return F.log_softmax(self.final_layer(x), dim=1)
        
    def train(model, train_dataloader, optimizer, device, print_freq=100):
        model.train()
        train_loss=0