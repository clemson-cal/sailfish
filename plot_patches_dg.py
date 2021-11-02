import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import msgpack

def patch_rect(patch):
    i0 = patch['rect'][0]['start']
    j0 = patch['rect'][1]['start']
    i1 = patch['rect'][0]['end']
    j1 = patch['rect'][1]['end']
    return i0, i1, j0, j1

def patch_size(patch):
    i0, i1, j0, j1 = patch_rect(patch)
    return i1 - i0, j1 - j0

def chkpt_num_poly(chkpt):
    sizeof_double = 8
    num_primitive = 4
    patch = chkpt['primitive_patches'][0]
    ni, nj = patch_size(patch)
    return len(patch['data']) // (ni * nj * sizeof_double * num_primitive)

fields = {'density': 0, 'px': 1, 'py': 2, 'energy': 3}
field_index = fields['density']

for filename in sys.argv[1:]:
    chkpt = msgpack.load(open(filename, 'rb'))
    mesh = chkpt['mesh']
    num_poly = chkpt_num_poly(chkpt)
    cons = np.zeros([mesh['ni'], mesh['nj'], 4 * num_poly])
    for patch in chkpt['primitive_patches']:
        i0, i1, j0, j1 = patch_rect(patch)
        local_prim = np.array(np.frombuffer(patch['data'])).reshape([i1 - i0, j1 - j0, 4 * num_poly])
        cons[i0:i1, j0:j1] = local_prim
    plt.figure()
    plt.imshow(cons[:, :, num_poly * field_index].T, origin='lower')

plt.show()
