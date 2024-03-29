import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(65536, 1024)
        self.fc21 = nn.Linear(1024, 64)
        self.fc22 = nn.Linear(1024, 64)
        self.fc3 = nn.Linear(64, 1024)
        self.fc4 = nn.Linear(1024, 65536)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, log_var = self.encode(x.view(-1, 65536))
        # print(mu, log_var)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
