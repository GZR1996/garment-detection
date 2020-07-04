import numpy as np
import matplotlib.pyplot as plt

c = np.load('./simulation/data/final_depth/0.0_0.0_0.0_1_0_0.npy')

a = np.load('./simulation/data/depth/0.0_0.0_0.0_1_0_0.npy')
b = np.load('./simulation/data/segmentation/0.0_0.0_0.0_1_0_0.npy')
c = np.load('./simulation/data/final_depth/0.0_0.0_0.0_1_0_0.npy')
np.savetxt('1.txt', a)
np.savetxt('2.txt', b)
d = a.copy()
d[(a != 1) & (b == 0)] = 1

plt.imshow(c)
plt.savefig('3.png')
plt.show()