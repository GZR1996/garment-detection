import torch
import os
import sys
import argparse

parser = argparse.ArgumentParser(description='Parameter of train.py')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint')
args = parser.parse_args()

if os.path.exists(args.checkpoint_dir):
    os.mkdir(args.checkpoint_dir)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

