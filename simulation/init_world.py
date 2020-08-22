import os
from collections import defaultdict
3
import numpy as np
import random

from simulation.environment import Environment, cal_parameter_group
from simulation.environment import DIRECTORY
from simulation.environment import DATA_FOLDER_NAME

# the parameter of cloth
table_position = np.array([0.0, 0.0, 0.0])

# the parameter of cloth
cloth_position = np.array([0.0, 0.0, 0.0])
cloth_orientation = np.array([0.0, 0.0, np.pi])

# simulation
environment = Environment(table_position, cloth_position)
# environment.simulate()

# generate labels
# for vae
file_list = os.listdir(os.path.join(DIRECTORY, DATA_FOLDER_NAME, 'bin'))
with open('./data/all.csv', 'w') as fin:
    fin.writelines((path.replace('.npz', '\n').replace('_', ',') for path in file_list))
random.shuffle(file_list)

boundary1 = int(0.8*len(file_list))
boundary2 = int(0.9*len(file_list))
path_list_train = [path.replace('.npz', '\n').replace('_', ',') for path in file_list[:boundary1]]
path_list_validate = [path.replace('.npz', '\n').replace('_', ',') for path in file_list[boundary1:boundary2]]
path_list_test = [path.replace('.npz', '\n').replace('_', ',') for path in file_list[boundary2:]]

if not os.path.exists('./data/vae'):
    os.mkdir('./data/vae')
with open('./data/vae/train_label.csv', 'w') as fin:
    fin.writelines(path_list_train)
with open('./data/vae/validate_label.csv', 'w') as fin:
    fin.writelines(path_list_validate)
with open('./data/vae/test_label.csv', 'w') as fin:
    fin.writelines(path_list_test)

file_list.insert(0, file_list[0])
file_dict = {}
for file in file_list:
    s = file.replace('.npz', '').split('_')
    if int(s[-3]) > 2:
        continue
    key = (s[0], s[1], s[2], s[4], s[5], s[6])
    if key not in file_dict:
        file_dict[key] = []
    file_dict[key].append(file)

label_dict = defaultdict(list)
for key in file_dict.keys():
    fl = sorted(file_dict[key], key=lambda x: int(x.replace('npz', '').split('_')[3]))
    for i in range(len(fl)-3):
        rand = random.randint(1, 100)
        line = '{},{},{}\n'.format(fl[i], fl[i+1], fl[i+2])
        if rand <= 75:
            label_dict['train'].append(line)
        elif rand <= 90:
            label_dict['test'].append(line)
        else:
            label_dict['validate'].append(line)

if not os.path.exists('./data/regression'):
    os.mkdir('./data/regression')
with open('./data/regression/train_label.csv', 'w') as fin:
    fin.writelines(label_dict['train'])
with open('./data/regression/validate_label.csv', 'w') as fin:
    fin.writelines(label_dict['test'])
with open('./data/regression/test_label.csv', 'w') as fin:
    fin.writelines(label_dict['validate'])


# for regression
# esr = environment.elastic_stiffness_range
# dsr = environment.damping_stiffness_range
# bsr = environment.bending_stiffness_range
# tr = range(0, 19, 3)
# cr = range(5)
#
# random.seed(123)
# label_dict = defaultdict(list)
# for parameter_count in range(0, 1000, 1):
#     indexes = cal_parameter_group(parameter_count)
#     es = esr[indexes[0]]
#     ds = dsr[indexes[1]]
#     bs = bsr[indexes[2]]
#
#     for c in cr:
#         for t in tr:
#             line = ''
#             for i in range(t, t+3):
#                 line += '{:.1f}_{:.1f}_{:.1f}_{:.0f}_{:.0f}.jpg'.format(es, ds, bs, i, c)
#                 if i != t+2:
#                     line += ','
#                 else:
#                     line += '\n'
#             rand = random.randint(1, 100)
#             if rand <= 75:
#                 label_dict['train'].append(line)
#             elif rand <= 90:
#                 label_dict['test'].append(line)
#             else:
#                 label_dict['validate'].append(line)
#
# if not os.path.exists('./data/regression'):
#     os.mkdir('./data/regression')
# with open('./data/regression/train_label.csv', 'w') as fin:
#     fin.writelines(label_dict['train'])
# with open('./data/regression/validate_label.csv', 'w') as fin:
#     fin.writelines(label_dict['test'])
# with open('./data/regression/test_label.csv', 'w') as fin:
#     fin.writelines(label_dict['validate'])
