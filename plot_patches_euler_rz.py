import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import msgpack

for filename in sys.argv[1:]:
    chkpt = msgpack.load(open(filename, 'rb'))
    mesh = chkpt['mesh']
    prim = np.zeros([mesh['ni'], mesh['nj'], 4])
    for patch in chkpt['primitive_patches']:
        i0 = patch['rect'][0]['start']
        j0 = patch['rect'][1]['start']
        i1 = patch['rect'][0]['end']
        j1 = patch['rect'][1]['end']
        local_prim = np.array(np.frombuffer(patch['data'])).reshape([i1 - i0, j1 - j0, 4])
        prim[i0:i1, j0:j1] = local_prim
    plt.imshow(prim[:,:,0].T, origin='lower')

plt.show()
