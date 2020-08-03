import torch.nn as nn
import torch
import torch.nn.functional as F


class Regression(nn.Module):
    """ Regression Network"""
    def __init__(self):
        super(Regression, self).__init__()
        self.unit1 = Net()
        self.unit2 = Net()
        self.unit3 = Net()
        self.fc1 = nn.Linear(3*32, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x1, x2, x3):
        y1 = self.unit1(x1)
        y2 = self.unit2(x2)
        y3 = self.unit3(x3)

        y = torch.cat((y1, y2, y3), 1)
        y = F.sigmoid(self.fc1(y))
        y = F.sigmoid(self.fc2(y))
        y = F.sigmoid(self.fc3(y))
        return y


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1, stride=2)
        self.fc1 = nn.Linear(8*16*16, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
