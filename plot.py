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
    x1 = chkpt['mesh']['dx'] * chkpt['mesh']['ni'] + x0
    y1 = chkpt['mesh']['dy'] * chkpt['mesh']['nj'] + y0
    for param in chkpt['parameters'].split(':'):
        print(param)
    for key, val in chkpt['command_line'].items():
        print(f'{key}: {val}')
    primitive = np.reshape(chkpt['primitive'], (ni + 4, nj + 4, 3))[2:-2,2:-2]
    plt.figure(figsize=[12, 9.5])
    plt.imshow(primitive[:,:,0].T**0.25, origin='lower', cmap='plasma', extent=[x0, x1, y0, y1])#, vmin=0.0, vmax=1.6)
    plt.xlim(-3.5, 3.5)
    plt.ylim(-3.5, 3.5)
    plt.colorbar()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.title(r"{} $\Sigma^{{1/4}}$".format(filename))

    plt.savefig(filename.replace('.sf', '.png'))
    plt.clf()
# plt.show()
