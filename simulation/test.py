import numpy as np
import matplotlib.pyplot as plt
import os

a = np.load('/home/zirun/project/python/garment-detection/simulation/data/depth/0.0_0.0_0.0_0_0.npy')
b = np.load('/home/zirun/project/python/garment-detection/simulation/data/segmentation/0.0_0.0_0.0_0_0.npy')
c = np.load('/home/zirun/project/python/garment-detection/simulation/data/final_depth/0.0_0.0_0.0_0_0.npy')
np.savetxt('1.txt', a)
np.savetxt('2.txt', b)
np.savez_compressed('da', a=a, b=b)

e = np.load('da.npz')
print(e['a'])
# d = a.copy()
# d[(a != 1) & (b == 0)] = 1
#
# plt.imshow(c)
# plt.savefig('3.png')
# plt.show()

