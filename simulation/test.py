import numpy as np
import matplotlib.pyplot as plt
import os

a = np.load('/home/zirun/project/python/garment-detection/simulation/data_10/bin/1200_0.1_0.1_5_2.npz')
b = np.load('/home/zirun/project/python/garment-detection/simulation/data_20/bin/0.1_0.1_0.1_5_2.npz')
c = np.load('/home/zirun/project/python/garment-detection/simulation/data_10/bin/1200_0.1_0.1_1_3.npy')
d = np.load('/home/zirun/project/python/garment-detection/simulation/data_20/bin/0.1_0.1_0.1_1_3.npy')

print(np.array_equal(a['depth'], b['depth']))
print(a['segmentation'])
e = np.load('da.npz')
print(e['a'])

q = a['depth']
w = b['depth']
flag = 0
for i in range(len(a['depth'])):
    for j in range(len(b['depth'])):
        if q[i][j] != w[i][j]:
            print(q[i][j], w[i][j])
            flag = 1
print("same" if flag == 0 else 1)
# d = a.copy()
# d[(a != 1) & (b == 0)] = 1
#
# plt.imshow(c)
# plt.savefig('3.png')
# plt.show()

