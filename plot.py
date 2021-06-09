import sys
import math
import numpy as np
import matplotlib.pyplot as plt

primitive = np.fromfile(sys.argv[1], dtype=np.float64)
n = math.isqrt(len(primitive) // 3)
primitive = primitive.reshape([n, n, 3])
plt.imshow(np.log10(primitive[:,:,0].T), origin='lower')
plt.colorbar()
plt.show()
