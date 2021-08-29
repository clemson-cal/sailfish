import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import msgpack

chkpt = msgpack.load(open(sys.argv[1], 'rb'))
time_series_data = np.array(chkpt['time_series_data'])
mdot1 = -1.0*time_series_data[:,0]
mdot2 = -1.0*time_series_data[:,3]

plt.plot(mdot1)
plt.plot(mdot2)

plt.show()
