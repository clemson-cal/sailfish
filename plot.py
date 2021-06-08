import sys
import math
import numpy as np
import matplotlib.pyplot as plt

primitive = np.fromfile(sys.argv[1])
n = math.isqrt(len(primitive) // 3)
primitive = primitive.reshape([n, n, 3])
plt.imshow(primitive[:,:,0])
plt.show()
