import numpy as np

a = np.load('./simulation/data/segmentation/0_0.npy')
b = np.load('./simulation/data/depth/0_0.npy')
np.savetxt("1.txt", a)
np.savetxt('2.txt', b)

c = np.ones_like(b)
c[(a == -1) & (b != 1.0)] = 0
# c[a == 1] = 0
np.savetxt("3.txt", c)
print(np.argwhere((b != 1.0)))

from PIL import Image

# image = Image.fromarray(a * 255)
# image = image.convert('L')
# image.save('1.png')
#
# image = Image.fromarray(b * 255)
# image = image.convert('L')
# image.save('2.png')
#
image = Image.fromarray(c * 255)
image = image.convert('RGB')
image.save('3.png')
