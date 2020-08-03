import argparse
import os
import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn import preprocessing
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision import transforms

from models.conv_regression import ConvRegression, TriEncoder, SimpleRegression
from models.regression import Regression, Net
from utils import utils
from utils.sample_dataset import SampleDataset
from utils.utils import EarlyStopping

DIRECTORY = os.path.abspath(os.path.dirname(__file__))
SAMPLE_DIR = os.path.join(DIRECTORY, 'simulation', 'data', 'bin')
TRAIN_LABEL_DIR = os.path.join(DIRECTORY, 'simulation', 'data', 'regression', 'train_label.csv')
TEST_LABEL_DIR = os.path.join(DIRECTORY, 'simulation', 'data', 'regression', 'test_label.csv')
VALIDATE_LABEL_DIR = os.path.join(DIRECTORY, 'simulation', 'data', 'regression', 'validate_label.csv')
ALL_DIR = os.path.join(DIRECTORY, 'simulation', 'data', 'all.csv')
CHECKPOINT_DIR = os.path.join(DIRECTORY, 'checkpoint', 'regression')
RESULT_DIR = os.path.join(DIRECTORY, 'result')
TARGET_MAP = {'elastic': 0, 'damping': 1, 'bending': 2}

parser = argparse.ArgumentParser(description='Parameters of train_regression.py')
parser.add_argument('--batch_size', type=int, default=32, help='The number of batches')
parser.add_argument('--epochs', type=int, default=10, help='The number of epochs')
parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR, help='Path to regression checkpoint')
parser.add_argument('--sample_dir', type=str, default=SAMPLE_DIR, help='Path to training or testing samples')
parser.add_argument('--result_dir', type=str, default=RESULT_DIR, help='Path to result')
parser.add_argument('--train_label_dir', type=str, default=TRAIN_LABEL_DIR, help='Path to train label')
parser.add_argument('--test_label_dir', type=str, default=TEST_LABEL_DIR, help='Path to test label')
parser.add_argument('--reload', type=int, default=1, choices=[0, 1],
                    help='If true and previous vae1 exists, reload the best vae1')
parser.add_argument('--generate_result', type=int, default=0, choices=[0, 1],
                    help='If true and previous regression checkpoint exists, generate prediction')
args = parser.parse_args()

if not os.path.exists(args.checkpoint_dir):
    os.mkdir(args.checkpoint_dir)
if not os.path.exists(args.result_dir):
    os.mkdir(args.result_dir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1000)


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def load_model(model_name, num_classes, feature_extract, use_pretrained=True):
    if model_name == 'regression':
        return 1
    elif model_name == 'vgg':
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_features = model_ft.classifier[6].in_features
        model_ft.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        model_ft.classifier[6] = nn.Linear(num_features, num_classes)
        input_size = 224
        return model_ft, input_size


def loss_function(x, recon_x, mu, log_sigma, outputs, targets):
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
    # print(recon_x.shape, x.shape)
    BCE = F.mse_loss(recon_x, x, size_average=False)
    MSE = F.mse_loss(outputs, targets, size_average=False)
    # KL divergence
    KLD = -0.5 * torch.sum(1 + 2 * log_sigma - mu.pow(2) - (2 * log_sigma).exp())
    loss = BCE + MSE + KLD

    return loss


def train(encoder, model, loader, criterion, stiffness, epoch):
    model.train()
    train_loss = 0.0
    epoch_start = time.time()
    label_encoer = label_encoders[stiffness]

    for batch, data in enumerate(loader):
        optimizer.zero_grad()
        x1 = data['x1'].to(device)
        x2 = data['x2'].to(device)
        x3 = data['x3'].to(device)
        z = encoder(x1, x2, x3)
        outputs = model(z)
        # outputs = torch.max(outputs, 1)# torch.tensor([torch.argmax(o) for o in outputs], dtype=torch.float, requires_grad=True).to(device)
        # print(batch, [np.argmax(o) for o in outputs.detach().cpu().numpy()])
        # print(batch, outputs.detach().cpu().numpy())
        # print(outputs)
        targets = label_encoer.transform([t for t in data['label'][:, stiffness]])
        targets = torch.tensor(targets, dtype=torch.long).to(device)
        loss = criterion(outputs, targets)
        # loss = torch.sqrt(loss)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch % 20 == 0:
            print('Train epoch: {}, batch: {}, loss: {}, time: {}'.format(epoch, batch, loss,
                                                                          time.time() - epoch_start))

    avg_loss = train_loss / len(loader.dataset)
    print('Finish training epoch {}, average loss: {}, in {} seconds'.format(epoch,
                                                                             avg_loss,
                                                                             time.time() - epoch_start))

    return avg_loss


def test(encoder, model, loader, criterion, stiffness, epoch, is_save=False):
    model.eval()
    test_loss = 0.0
    epoch_start = time.time()
    label_encoder = label_encoders[stiffness]
    accuracy = 0
    if is_save:
        results = []
        true_labels = []

    with torch.no_grad():
        for batch, data in enumerate(loader):
            x1 = data['x1'].to(device)
            x2 = data['x2'].to(device)
            x3 = data['x3'].to(device)
            z = encoder(x1, x2, x3)
            outputs = model(z)
            # outputs = torch.max(outputs, 1) # outputs = torch.tensor([torch.argmax(o) for o in outputs], dtype=torch.float, requires_grad=True).to(device)
            targets = label_encoder.transform([t for t in data['label'][:, stiffness]])
            targets_ = torch.tensor(targets, dtype=torch.long).to(device)
            loss = criterion(outputs, targets_)
            # loss = torch.sqrt(loss)
            test_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            accuracy += torch.sum(preds == targets_)
            if is_save:
                true_labels.extend([t for t in targets])
                results.extend(preds.cpu().numpy())
            if batch % 20 == 0:
                print('Test epoch: {}, batch: {}, loss: {}, time: {}'.format(epoch, batch, loss,
                                                                             time.time() - epoch_start))
                # print('Example outputs: ', preds, 'True label: ', targets)

    avg_loss = test_loss / len(loader.dataset)
    if is_save:
        print(classification_report(true_labels, results))
        print(results)
        results = label_encoder.inverse_transform(results)

        np.savetxt(os.path.join(args.result_dir, 'result.csv'), results, delimiter=',')
    else:
        print('Finish testing epoch {}, average loss: {}, accuracy: {}, in {} seconds'.format(epoch,
                                                                                              avg_loss,
                                                                                              accuracy / len(
                                                                                                  loader.dataset),
                                                                                              time.time() - epoch_start))

    return avg_loss


elastic_stiffness_range = np.arange(100.0, 1500.0, 300.0)
damping_stiffness_range = np.arange(0.1, 1.1, 0.1)
bending_stiffness_range = np.arange(2.0, 22.0, 2.0)

elastic_le = preprocessing.LabelEncoder()
elastic_le.fit(elastic_stiffness_range)
print(elastic_le.classes_)
damping_le = preprocessing.LabelEncoder()
damping_le.fit(damping_stiffness_range)
bending_le = preprocessing.LabelEncoder()
bending_le.fit(bending_stiffness_range)
label_encoders = [elastic_le, damping_le, bending_le]

model = SimpleRegression(3 * 32)
# model = ConvRegression(utils.DATA_SIZE, utils.LATENT_SIZE)
encoder = TriEncoder(1, 32).to(device)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
earlystopping = EarlyStopping('min', patience=30)
loss_function = nn.CrossEntropyLoss()

data_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

train_dataset = SampleDataset(args.sample_dir, args.train_label_dir)
test_dataset = SampleDataset(args.sample_dir, args.test_label_dir)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)

best_loss = None
stiffness = 'elastic'
target_stiffness = TARGET_MAP[stiffness]
checkpoint_count = len(os.listdir(args.checkpoint_dir))
reload_dir = os.path.join(args.checkpoint_dir, utils.BEST_FILENAME)

# if len(os.listdir(args.checkpoint_dir)) == 0:
#     best_state = torch.load('./checkpoint/vae/best.pth', map_location='cpu')
#     model.encoder1.load_state_dict(best_state['encoder_dict'])
#     model.encoder2.load_state_dict(best_state['encoder_dict'])
#     model.encoder3.load_state_dict(best_state['encoder_dict'])
#     # model.decoder.load_state_dict(best_state['decoder_dict'])
#     # model.encoder1.load_state_dict(best_state[''])
#     del best_state
#     os._exit(0)
# else:
# encoder_state = torch.load('./checkpoint/vae/best.pth', map_location='cpu')
# encoder.encoder.load_state_dict(encoder_state['encoder_dict'])
# del encoder_state
if args.generate_result == 0 and args.reload == 1 and os.path.exists(reload_dir):
    best_state = torch.load(reload_dir)
    print('Reloading vae1......, file: ', reload_dir)
    model.load_state_dict(best_state['state_dict'])
    optimizer.load_state_dict(best_state['optimizer_dict'])
    scheduler.load_state_dict(best_state['scheduler_dict'])
    earlystopping.load_state_dict(best_state['earlystopping_dict'])
    # delete useless parameter to get more gpu memory
    del best_state

# generate result
if args.generate_result == 1:
    reload_dir = os.path.join(args.checkpoint_dir, utils.BEST_FILENAME)
    best_state = torch.load(reload_dir, map_location='cpu')
    print('Loading the best vae1......')
    print('Start generate samples......')
    model.load_state_dict(best_state['state_dict'])
    # delete useless parameter to get more gpu memory
    del best_state
    # test(vae1, train_loader, 0, is_save=True)
    validate_dataset = SampleDataset(args.sample_dir, args.test_label_dir)
    validate_loader = DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    test(encoder, model, validate_loader, loss_function, target_stiffness, 0, is_save=True)
    os._exit(0)

for epoch in range(args.epochs):
    train_loss = train(encoder, model, train_loader, loss_function, target_stiffness, epoch)
    test_loss = test(encoder, model, test_loader, loss_function, target_stiffness, epoch)
    scheduler.step(train_loss)
    earlystopping.step(train_loss)
    is_best = not best_loss or test_loss < best_loss
    if is_best:
        best_loss = test_loss

    loss_state = {'epoch': checkpoint_count,
                  'train_loss': train_loss,
                  'test_loss': test_loss}
    best_state = {'epoch': checkpoint_count,
                  'state_dict': model.state_dict(),
                  'optimizer_dict': optimizer.state_dict(),
                  'scheduler_dict': scheduler.state_dict(),
                  'earlystopping_dict': earlystopping.state_dict(),
                  'train_loss': train_loss,
                  'test_loss': test_loss,
                  'best_loss': best_loss}
    checkpoint_name = os.path.join(args.checkpoint_dir, str(checkpoint_count) + '.pth')
    utils.save_checkpoint(loss_state, best_state, is_best, checkpoint_name, reload_dir)
    checkpoint_count += 1
