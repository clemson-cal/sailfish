import sys
import math
import numpy as np
import matplotlib.pyplot as plt

primitive = np.fromfile(sys.argv[1], dtype=np.float64)
n = math.isqrt(len(primitive) // 3)
primitive = primitive.reshape([n, n, 3])
plt.figure(figsize=[12, 9.5])
plt.imshow(primitive[:,:,0].T**0.25, origin='lower', cmap='plasma', extent=[-8, 8, -8, 8])
# plt.xlim(-4, 4)
# plt.ylim(-4, 4)
plt.colorbar()
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
plt.title(r"$\Sigma^{1/4}$")
plt.show()
# plt.savefig('sailfish.pdf')
