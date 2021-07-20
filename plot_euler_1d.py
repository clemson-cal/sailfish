import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import msgpack

for filename in sys.argv[1:]:
    chkpt = msgpack.load(open(filename, 'rb'))
    faces = np.array(chkpt['mesh'])
    prims = np.array(chkpt['primitive']).reshape([len(faces) - 1, 3])
    fig = plt.figure(figsize=[10, 8])
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.stairs(prims[:,0], faces, label=r'$\rho$')
    ax1.stairs(prims[:,2], faces, label=r'$p$')

ax1.legend()
plt.show()
