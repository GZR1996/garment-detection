import argparse
import os
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from models.vae import VAE
from utils.garment_dataset import GarmentDataset
from utils.utils import DATA_SIZE, LATENT_SIZE, BEST_FILENAME
from utils.utils import save_checkpoint

# parameters of training
parser = argparse.ArgumentParser(description='Parameter of train.py')
parser.add_argument('--batch_size', type=int, default=16, help='The number of batches')
parser.add_argument('--epochs', type=int, default=10, help='The number of epochs for training and testing')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='Path to checkpoint folder')
parser.add_argument('--data_dir', type=str, default='./simulation/data/', help='Path to data folder')
parser.add_argument('--reload', type=bool, default=True,
                    help='If true and previous checkpoint exists, reload the best checkpoint')
args = parser.parse_args()

if os.path.exists(args.checkpoint_dir):
    os.mkdir(args.checkpoint_dir)
else:
    best_checkpoint = torch.load(os.path.join(args.chepoint_dir, BEST_FILENAME))
rgb_dir = os.path.join(parser.data_dir, 'rgb')
data_dir = os.path.join(parser.data_dir, 'bin')
train_label_dir = os.path.join(parser.data_dir, 'train_label.csv')
test_label_dir = os.path.join(parser.data_dir, 'test_label.csv')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1000)


def loss_function(recon_x, x, mu, log_sigma):
    """
    VAE loss function
    :param recon_x: reconstructed image
    :param x: image
    :param mu: mean of latent factor
    :param log_sigma: standard deviation of latent factor
    :return: loss
    """
    # MSE for batch
    BCE = F.mse_loss(recon_x, x, size_average=False)
    # KL divergence
    KLD = -0.5 * torch.sum(1 + 2 * log_sigma - mu.pow(2) - (2 * log_sigma).exp())
    loss = BCE + KLD

    return loss


def train(model, loader, epoch):
    model.train()
    train_loss = 0.0
    epoch_start = time.time()

    for batch, data in enumerate(loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, sigma = model(data)
        loss = loss_function(recon_batch, data, mu, sigma)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch % 20 == 0:
            print('Train epoch: {}, batch: {}, loss: {}, time: {}'.format(epoch, batch, loss, time.time()-epoch_start))

    avg_loss = train_loss / len(loader.dataset)
    print('Finish training epoch {}, average loss: {}, in {} seconds'.format(epoch,
                                                                             avg_loss,
                                                                             time.time()-epoch_start))

    return avg_loss


def test(model, loader, epoch):
    model.eval()
    test_loss = 0
    epoch_start = time.time()

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            recon_batch, mu, sigma = model(data)
            test_loss += loss_function(recon_batch, data, mu, sigma)

    avg_loss = test_loss / len(loader.dataset)
    print('Finish training epoch {}, average loss: {}, in {} seconds'.format(epoch,
                                                                             avg_loss,
                                                                             time.time() - epoch_start))

    return avg_loss


train_dataset = GarmentDataset(data_dir, parser.data_type, train_label_dir)
test_dataset = GarmentDataset(data_dir, parser.data_type, test_label_dir)
train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8)
test_loader = DataLoader(test_dataset, args.batch_size, shuffle=True, num_workers=8)

vae = VAE(DATA_SIZE, LATENT_SIZE).to(device)
optimizer = optim.Adam(vae.parameters())

reload_dir = os.path.join(parser.checkpoint_dir, BEST_FILENAME)
checkpoint_count = len(os.listdir(reload_dir))
if parser.reload and len(os.listdir(parser.checkpoint_dir)) > 0:
    state = torch.load(reload_dir)
    print('Reloading checkpoint......')
    vae.load_state_dict(state['state_dict'])

best_loss = None
for epoch in range(parser.epochs):
    train_loss = train(vae, train_loader, epoch)
    test_loss = test(vae, test_loader, epoch)
    is_best = not best_loss or test_loss < best_loss
    if is_best:
        best_loss = test_loss

    state = {'epoch': checkpoint_count+epoch,
             'state_dict': vae.state_dict(),
             'train_loss': train_loss,
             'test_loss': test_loss}
    checkpoint_name = os.path.join(parser.checkpoint_dir, str(checkpoint_count)+'.txt')
    save_checkpoint(state, is_best, checkpoint_name, reload_dir)

