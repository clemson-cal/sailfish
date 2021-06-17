import sys
import math
import numpy as np
import matplotlib.pyplot as plt

for filename in sys.argv[1:]:
    primitive = np.fromfile(filename, dtype=np.float64)
    n = math.isqrt(len(primitive) // 3)
    primitive = primitive.reshape([n, n, 3])
    plt.figure(figsize=[12, 9.5])
    plt.imshow(primitive[:,:,0].T, origin='lower', cmap='plasma')
    plt.colorbar()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.title(r"{} $\Sigma^{{1/4}}$".format(filename))

plt.show()
