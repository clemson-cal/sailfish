import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import msgpack
import glob

for filename in sorted(glob.glob("/home/binayyr/bondi/temp_data/data/binarybondi_run_final1/chkpt.****.sf")):
  fig = plt.figure()
  ax1 = fig.add_subplot(1,1,1)
  fig.set_size_inches(20.0, 20.0)
  chkpt = msgpack.load(open(filename, 'rb'))
  mesh = chkpt['mesh']
  print("1"+filename)
  prim = np.zeros([mesh['ni'], mesh['nj'], 3])
  for patch in chkpt['primitive_patches']:
    i0 = patch['rect'][0]['start']
    j0 = patch['rect'][1]['start']
    i1 = patch['rect'][0]['end']
    j1 = patch['rect'][1]['end']
    local_prim = np.array(np.frombuffer(patch['data'])).reshape([i1 - i0, j1 - j0, 3])
    print("2"+filename)
    prim[i0:i1, j0:j1] = local_prim
  #print(filename)
  cm = ax1.imshow(prim[:,:,0].T, origin='lower',vmin=0.0,vmax=10.0, cmap='magma')
  cbar = fig.colorbar(cm,ax=ax1)
  cbar.ax.tick_params(labelsize=20)
  ax1.tick_params(axis='y', which='major', labelsize=20)
  ax1.tick_params(axis='x', which='major', labelsize=20)
  fig.savefig(filename+".png")
  fig.clf()
  #plt.close("all")
