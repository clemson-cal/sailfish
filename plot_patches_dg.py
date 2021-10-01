import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import msgpack

num_poly = 3

for filename in sys.argv[1:]:
    chkpt = msgpack.load(open(filename, 'rb'))
    mesh = chkpt['mesh']
    prim = np.zeros([mesh['ni'], mesh['nj'], 4 * num_poly])
    for patch in chkpt['primitive_patches']:
        # print(patch['rect'])
        i0 = patch['rect'][0]['start']
        j0 = patch['rect'][1]['start']
        i1 = patch['rect'][0]['end']
        j1 = patch['rect'][1]['end']
        local_prim = np.array(np.frombuffer(patch['data'])).reshape([i1 - i0, j1 - j0, 4 * num_poly])
        prim[i0:i1, j0:j1] = local_prim
    plt.imshow(np.log10(prim[:,:,0].T), origin='lower')

plt.show()
