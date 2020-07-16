import numpy as np
from sklearn import preprocessing

import argparse
import os

DIRECTORY = os.path.abspath(os.path.dirname(__file__))
SAMPLE_DIR = os.path.join(DIRECTORY, 'simulation', 'data', 'sample')
TRAIN_LABEL_DIR = os.path.join(DIRECTORY, 'simulation', 'data', 'train_label.csv')
TEST_LABEL_DIR = os.path.join(DIRECTORY, 'simulation', 'data', 'test_label.csv')
CHECKPOINT_DIR = os.path.join(DIRECTORY, 'checkpoint', 'vae')

parser = argparse.ArgumentParser(description='Parameters of train_regression.py')
parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR, )

elastic_stiffness_range = np.arange(40.0, 140.0, 10.0)
damping_stiffness_range = np.arange(0.1, 1.1, 0.1)
bending_stiffness_range = np.arange(2.0, 22.0, 2.0)

elastic_le = preprocessing.LabelEncoder()
elastic_le.fit(elastic_stiffness_range)
damping_le = preprocessing.LabelEncoder()
damping_le.fit(damping_stiffness_range)
bending_le = preprocessing.LabelEncoder()
bending_le.fit(bending_stiffness_range)

