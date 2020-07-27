import torch
import torch.nn as nn
import torch.nn.functional as F


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class TriEncoder(nn.Module):

    def __init__(self, img_channels, latent_size):
        super(TriEncoder, self).__init__()
        self.encoder1 = Encoder(img_channels, latent_size)
        self.encoder2 = Encoder(img_channels, latent_size)
        self.encoder3 = Encoder(img_channels, latent_size)

    def forward(self, x1, x2, x3):
        mu1, log_sigma1 = self.encoder1(x1)
        mu2, log_sigma2 = self.encoder2(x2)
        mu3, log_sigma3 = self.encoder3(x3)

        sigma = torch.cat((log_sigma1.exp(), log_sigma2.exp(), log_sigma3.exp()), 1)
        mu = torch.cat((mu1, mu2, mu3), 1)
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)
        return z

class ConvRegression(nn.Module):
    """ VAE """

    def __init__(self, img_channels, latent_size):
        super(ConvRegression, self).__init__()
        self.encoder1 = Encoder(img_channels, latent_size)
        self.encoder2 = Encoder(img_channels, latent_size)
        self.encoder3 = Encoder(img_channels, latent_size)
        self.decoder = Decoder(img_channels, latent_size)
        # self.regression = Regression(img_channels, latent_size)

    def forward(self, x1, x2, x3):
        mu1, log_sigma1 = self.encoder1(x1)
        mu2, log_sigma2 = self.encoder2(x2)
        mu3, log_sigma3 = self.encoder3(x3)

        sigma = torch.cat((log_sigma1.exp(), log_sigma2.exp(), log_sigma3.exp()), 1)
        mu = torch.cat((mu1, mu2, mu3), 1)
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)

        recon_x, output = self.decoder(z)
        # output = self.regression(z)
        return recon_x, mu, sigma, output


class Encoder(nn.Module):
    """ VAE encoder """

    def __init__(self, img_channels, latent_size):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels

        self.conv1 = nn.Conv2d(img_channels, 32, 4, padding=True, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, padding=True, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, padding=True, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, padding=True, stride=2)

        self.fc_mu = nn.Linear(256 * 16 * 16, latent_size)
        self.fc_log_sigma = nn.Linear(256 * 16 * 16, latent_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        log_sigma = self.fc_log_sigma(x)
        return mu, log_sigma


class Decoder(nn.Module):
    """ VAE decoder as a regression"""

    def __init__(self, img_channels, latent_size):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels

        self.fc1 = nn.Linear(3*latent_size, 64*16*16)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 4, padding=True, stride=2)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 4, padding=True, stride=2)
        self.deconv3 = nn.ConvTranspose2d(16, 8, 4, padding=True, stride=2)
        self.deconv4 = nn.ConvTranspose2d(8, 3, 4, padding=True, stride=2)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(4096, 256)
        self.fc5 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.view([-1, 64, 16, 16])
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        reconstruction = F.sigmoid(self.deconv4(x))
        # print(reconstruction.shape)
        x = F.relu(self.fc2(reconstruction))
        x = F.relu(self.fc3(x))
        x = torch.flatten(x, start_dim=2)
        x = F.relu(self.fc4(x))
        x = F.sigmoid(self.fc5(x))

        return reconstruction, x


class Regression(nn.Module):

    def __init__(self, img_channels, latent_size):
        super(Regression, self).__init__()
        # self.conv1 = nn.Conv2d(img_channels, 4, 3, padding=1, stride=2)
        # self.conv2 = nn.Conv2d(4, 8, 3, padding=1, stride=2)
        # self.fc1 = nn.Linear(8 * 16 * 16, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

        self.fc1 = nn.Linear(3*latent_size, 256*16*16)
        self.deconv1 = nn.ConvTranspose2d(256, 64, 4, padding=True, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, padding=True, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, padding=True, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, img_channels, 4, padding=True, stride=2)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # x = F.max_pool2d(x, (2, 2))
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = x.view(-1, num_flat_features(x))
        # x = F.relu(self.fc1(x))
        # x = F.sigmoid(self.fc2(x))
        # x = F.sigmoid(self.fc3(x))

        x = F.relu(self.fc1(x))
        x = x.view([-1, 256, 16, 16])
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.sigmoid(self.deconv4(x))
        x = F.sigmoid(self.fc2(x))
        return x
