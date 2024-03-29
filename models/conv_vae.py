import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    """ VAE """

    def __init__(self, img_channels, latent_size):
        super(ConvVAE, self).__init__()
        self.encoder = Encoder(img_channels, latent_size)
        self.decoder = Decoder(img_channels, latent_size)

    def forward(self, x):
        mu, log_sigma = self.encoder(x)
        sigma = log_sigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)

        recon_x = self.decoder(z)
        return recon_x, mu, sigma


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

        self.fc_mu = nn.Linear(256*16*16, latent_size)
        self.fc_log_sigma = nn.Linear(256*16*16, latent_size)

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
    """ VAE decoder"""

    def __init__(self, img_channels, latent_size):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels

        self.fc1 = nn.Linear(latent_size, 256*16*16)
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, padding=True, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, padding=True, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, padding=True, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, img_channels, 4, padding=True, stride=2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.view([-1, 256, 16, 16])
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        reconstruction = F.sigmoid(self.deconv4(x))

        return reconstruction
