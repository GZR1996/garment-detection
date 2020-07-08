import numpy as np
import matplotlib.pyplot as plt
import os

c = np.load('/home/zirun/project/python/garment-detection/simulation/data/bin/40.0_0.1_20.0_5_0.npz')
d = np.load('/home/zirun/project/python/garment-detection/simulation/data/bin/40.0_0.1_2.0_5_0.npz')

print("same" if np.array_equal(c['depth'], d['raw_depth']) else 0)
print(np.arange(0.1, 1, 0.1))
print("{:.2f}".format(1.23467525))
# d = a.copy()
# d[(a != 1) & (b == 0)] = 1
#
# plt.imshow(c)
# plt.savefig('3.png')
# plt.show()

