import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from models.conv_vae import ConvVAE
from utils.garment_dataset import GarmentDataset
from utils import utils

# Constants
from utils.utils import EarlyStopping

DIRECTORY = os.path.abspath(os.path.dirname(__file__))
CHECKPOINT_DIR = os.path.join(DIRECTORY, 'checkpoint', 'vae')
DATA_DIR = os.path.join(DIRECTORY, 'simulation', 'data', 'bin')
RGB_DIR = os.path.join(DIRECTORY, 'simulation', 'data', 'rgb')
SAMPLE_DIR = os.path.join(DIRECTORY, 'simulation', 'data', 'sample')
TRAIN_LABEL_DIR = os.path.join(DIRECTORY, 'simulation', 'data', 'train_label.csv')
TEST_LABEL_DIR = os.path.join(DIRECTORY, 'simulation', 'data', 'test_label.csv')

# parameters of training
parser = argparse.ArgumentParser(description='Parameter of train_vae.py')
parser.add_argument('--batch_size', type=int, default=32, help='The number of batches')
parser.add_argument('--epochs', type=int, default=10, help='The number of epochs for training and testing')
parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR, help='Path to vae folder')
parser.add_argument('--rgb_dir', type=str, default=RGB_DIR, help='Path to img folder')
parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='Path to data folder')
parser.add_argument('--data_type', type=str, choices=['rgb', 'raw_depth', 'depth', 'segmentation'], default='depth',
                    help='The data type of data')
parser.add_argument('--sample_dir', type=str, default=SAMPLE_DIR, help='Path to sample folder')
parser.add_argument('--train_label_dir', type=str, default=TRAIN_LABEL_DIR, help='Path to train label')
parser.add_argument('--test_label_dir', type=str, default=TRAIN_LABEL_DIR, help='Path to test label')
parser.add_argument('--reload', type=int, default=1, choices=[0, 1],
                    help='If true and previous vae exists, reload the best vae')
parser.add_argument('--generate_sample', type=int, choices=[0, 1], default=0,
                    help='If true, the model will not be trained and only generate samples')
args = parser.parse_args()

if not os.path.exists(args.checkpoint_dir):
    os.mkdir(args.checkpoint_dir)
if not os.path.exists(args.sample_dir):
    os.mkdir(args.sample_dir)

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
    # recon_x = recon_x.view([-1, 1, 256, 256])
    BCE = F.mse_loss(recon_x, x, size_average=False)
    # KL divergence
    KLD = -0.5 * torch.sum(1 + 2 * log_sigma - mu.pow(2) - (2 * log_sigma).exp())
    loss = BCE + KLD

    return loss


def train(model, loader, epoch):
    """ Train the model """
    model.train()
    train_loss = 0.0
    epoch_start = time.time()

    for batch, data in enumerate(loader):
        sample = data['sample'].to(device)
        optimizer.zero_grad()
        recon_batch, mu, sigma = model(sample)
        loss = loss_function(recon_batch, sample, mu, sigma)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch % 20 == 0:
            print(
                'Train epoch: {}, batch: {}, loss: {}, time: {}'.format(epoch, batch, loss, time.time() - epoch_start))

    avg_loss = train_loss / len(loader.dataset)
    print('Finish training epoch {}, average loss: {}, in {} seconds'.format(epoch,
                                                                             avg_loss,
                                                                             time.time() - epoch_start))

    return avg_loss


def test(model, loader, epoch, is_save=False):
    """ Test the model """
    model.eval()
    test_loss = 0
    epoch_start = time.time()

    with torch.no_grad():
        for batch, data in enumerate(loader):
            sample = data['sample'].to(device)
            recon_batch, mu, sigma = model(sample)
            test_loss += loss_function(recon_batch, sample, mu, sigma)
            if is_save:
                if batch % 20 == 0:
                    print('Saving {} batch images in {} seconds'.format(batch, time.time() - epoch_start))
                images = recon_batch.view([-1, 256, 256])
                utils.save_image(args.sample_dir, np.asarray(images.to('cpu')), np.asarray(data['label']))

    avg_loss = test_loss / len(loader.dataset)
    if not is_save:
        print('Finish testing epoch {}, average loss: {}, in {} seconds'.format(epoch,
                                                                                avg_loss,
                                                                                time.time() - epoch_start))
    else:
        print('Finish generate final result at', args.sample_dir)

    return avg_loss


data_transforms = transforms.Compose([transforms.RandomResizedCrop(256),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train_dataset = GarmentDataset(args.rgb_dir, args.data_dir, args.data_type, args.train_label_dir)
test_dataset = GarmentDataset(args.rgb_dir, args.data_dir, args.data_type, args.test_label_dir)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)

vae = ConvVAE(img_channels=utils.DATA_SIZE, latent_size=utils.LATENT_SIZE).to(device)
optimizer = optim.Adam(vae.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
earlystopping = EarlyStopping('min', patience=30)

checkpoint_count = len(os.listdir(args.checkpoint_dir))
reload_dir = os.path.join(args.checkpoint_dir, utils.BEST_FILENAME)
if args.generate_sample == 0 and args.reload == 1 and os.path.exists(reload_dir):
    best_state = torch.load(reload_dir)
    print('Reloading vae......, file: ', reload_dir)
    vae.load_state_dict(best_state['state_dict'])
    optimizer.load_state_dict(best_state['optimizer_dict'])
    scheduler.load_state_dict(best_state['scheduler_dict'])
    earlystopping.load_state_dict(best_state['earlystopping_dict'])
    # delete useless parameter to get more gpu memory
    del best_state

# generate result
if args.generate_sample == 1:
    reload_dir = os.path.join(args.checkpoint_dir, utils.BEST_FILENAME)
    best_state = torch.load(reload_dir)
    print('Loading the best vae......')
    print('Start generate samples......')
    vae.load_state_dict(best_state['state_dict'])
    # delete useless parameter to get more gpu memory
    del best_state
    test_loss = test(vae, test_loader, 0, is_save=True)
    os._exit(0)

best_loss = None
for epoch in range(args.epochs):
    train_loss = train(vae, train_loader, checkpoint_count)
    scheduler.step(train_loss)
    earlystopping.step(train_loss)
    test_loss = test(vae, test_loader, checkpoint_count)
    is_best = not best_loss or test_loss < best_loss
    if is_best:
        best_loss = test_loss

    loss_state = {'epoch': checkpoint_count,
                  'train_loss': train_loss,
                  'test_loss': test_loss}
    best_state = {'epoch': checkpoint_count,
                  'state_dict': vae.state_dict(),
                  'optimizer_dict': optimizer.state_dict(),
                  'scheduler_dict': scheduler.state_dict(),
                  'earlystopping_dict': earlystopping.state_dict(),
                  'train_loss': train_loss,
                  'test_loss': test_loss}
    checkpoint_name = os.path.join(args.checkpoint_dir, str(checkpoint_count) + '.pth')
    utils.save_checkpoint(loss_state, best_state, is_best, checkpoint_name, reload_dir)
    checkpoint_count += 1
