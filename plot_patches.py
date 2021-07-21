import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import msgpack

for filename in sys.argv[1:]:
    chkpt = msgpack.load(open(filename, 'rb'))
    mesh = chkpt['mesh']
    prim = np.zeros([mesh['ni'], mesh['nj'], 3])
    for patch in chkpt['primitive_patches']:
        i0 = patch['rect'][0]['start']
        j0 = patch['rect'][1]['start']
        i1 = patch['rect'][0]['end']
        j1 = patch['rect'][1]['end']
        local_prim = np.array(patch['data']).reshape([i1 - i0, j1 - j0, 3])
        prim[i0+2:i1-2, j0+2:j1-2] = local_prim[2:-2, 2:-2]
        print(i0, i1)
    plt.imshow(prim[:,:,0])

plt.show()
