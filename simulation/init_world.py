import os
import numpy as np
import random

from simulation.environment import Environment
from simulation.environment import DIRECTORY
from simulation.environment import DATA_FOLDER_NAME

# the parameter of cloth
table_position = np.array([0, -0.75, 0])

# the parameter of cloth
cloth_position = np.array([0, -1, 0.65])
cloth_orientation = np.array([0, 0, np.pi])
cloth_color = np.array([1, 0.5, 0.5, 1])
cloth_line_color = np.array([0, 0.5, 0, 1])

# simulation
environment = Environment(table_position, cloth_position)
environment.simulate()

# generate labels
file_list = os.listdir(os.path.join(DIRECTORY, DATA_FOLDER_NAME, 'bin'))
random.shuffle(file_list)

boundary = int(0.8*len(file_list))
path_list_train = [path.replace('.npz', '\n').replace('_', ',') for path in file_list[:boundary]]
path_list_test = [path.replace('.npz', '\n').replace('_', ',') for path in file_list[boundary:]]

with open('./data/label_train.csv', 'w') as fin:
    fin.writelines(path_list_train)
with open('./data/label_test.csv', 'w') as fin:
    fin.writelines(path_list_test)
