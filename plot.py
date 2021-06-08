import sys
import math
import numpy as np
import matplotlib.pyplot as plt

primitive = np.fromfile(sys.argv[1])
n = math.isqrt(len(primitive) // 3)
primitive = primitive.reshape([n, n, 3])
# plt.plot(primitive[n//2,:,0])
plt.imshow(primitive[:,:,0].T, origin='lower')
plt.axvline(n // 2 - 0.5)
plt.axhline(n // 2 - 0.5)
plt.colorbar()
plt.show()
