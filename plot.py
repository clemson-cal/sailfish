import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import msgpack

for filename in sys.argv[1:]:
    chkpt = msgpack.load(open(filename, 'rb'))
    ni = chkpt['mesh']['ni']
    nj = chkpt['mesh']['nj']
    x0 = chkpt['mesh']['x0']
    y0 = chkpt['mesh']['y0']
    x1 = chkpt['mesh']['dx'] + x0
    y1 = chkpt['mesh']['dy'] + y0
    primitive = np.reshape(chkpt['primitive'], (ni + 4, nj + 4, 3))[2:-2,2:-2]
    plt.figure(figsize=[12, 9.5])
    plt.imshow(primitive[:,:,0].T**0.25, origin='lower', cmap='plasma', extent=[x0, x1, y0, y1])
    plt.colorbar()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.title(r"{} $\Sigma^{{1/4}}$".format(filename))

plt.show()
