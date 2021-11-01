import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import msgpack

plt.figure(figsize=(8, 12), dpi=150)

for i in range(500):
    if i > 99:
        filename = 'chkpt.0%d.sf' %i
    elif i > 9:
        filename = 'chkpt.00%d.sf' %i
    else:
        filename = 'chkpt.000%d.sf' %i
    chkpt = msgpack.load(open(filename, 'rb'))
    mesh = chkpt['mesh']
    prim = np.zeros([mesh['ni'], mesh['nj'], 5])
    for patch in chkpt['primitive_patches']:
        i0 = patch['rect'][0]['start']
        j0 = patch['rect'][1]['start']
        i1 = patch['rect'][0]['end']
        j1 = patch['rect'][1]['end']
        local_prim = np.array(np.frombuffer(patch['data'])).reshape([i1 - i0, j1 - j0, 5])
        prim[i0:i1, j0:j1] = local_prim
    plt.imshow(prim[:,:,4].T, origin='lower', extent=[0, 1, -1, 1], vmin=0.0, vmax=1.0)
    plt.savefig('figures/Figure%d.png' %i)
    plt.clf()
