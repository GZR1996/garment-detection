import os
import pandas
import numpy as np
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

environment = Environment(table_position, cloth_position)
environment.simulate()
# environment.test()

# generate labels
data_dir = sorted(os.listdir(os.path.join(DIRECTORY, DATA_FOLDER_NAME, 'bin')))
path_list = []
for path in data_dir:
    parameters = path.replace('.npz', '\n').replace('_', ',')
    print(parameters)
    path_list.append(parameters)
with open('./data/label.csv', 'w') as fin:
    fin.writelines(path_list)