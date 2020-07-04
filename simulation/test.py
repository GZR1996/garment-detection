import numpy as np
import imageio
from PIL import Image
import matplotlib.pyplot as plt

c = np.load('./simulation/data/final_depth/0.0_0.0_0.0_1_0_0.npy')

a = np.load('./simulation/data/depth/0.0_0.0_0.0_1_0_0.npy')
b = np.load('./simulation/data/segmentation/0.0_0.0_0.0_1_0_0.npy')
c = np.load('./simulation/data/final_depth/0.0_0.0_0.0_1_0_0.npy')
np.savetxt('1.npy', c)
np.savetxt('2.npy', b)
d = a.copy()
d[(a != 1) & (b == 0)] = 1

plt.imshow(c)
# plt.savefig('3.png')
plt.show()
# image = Image.fromarray(a * 255)
# image = image.convert('L')
# image.save('1.png')
#
# image = Image.fromarray(b * 255)
# image = image.convert('L')
# image.save('2.png')
#
# image = Image.fromarray(c * 255)
# image = image.convert('L')
# image.save('3.png')