from sys import path
from argparse import ArgumentParser
from pickle import load
from numpy import array
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

path.insert(1, ".")

parser = ArgumentParser()
parser.add_argument("checkpoint", type=str)
parser.add_argument("x", type=int, default=0)
parser.add_argument("y", type=str, default=1, nargs="+")
parser.add_argument("--smooth", type=int, default=None)
args = parser.parse_args()

fig = plt.figure(figsize=[8, 6])
ax1 = fig.add_subplot(111)

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

with open(args.checkpoint, "rb") as f:
    chkpt = load(f)
    simulation_time_series = chkpt["timeseries"]

for key, c in zip(args.y, colors):
    x = array([t[args.x] for t in simulation_time_series])
    i = int(key)
    y = [t[i] for t in simulation_time_series]
    label = str(chkpt["diagnostics"][i])

    if args.smooth:
        y = savgol_filter(y, args.smooth, 3)

    ax1.plot(x, y, label=label, c=c)
    ax1.set_xlabel(f"{chkpt['diagnostics'][args.x]}")

ax1.grid(alpha=0.2)
fig.legend()
plt.show()
