from sys import path
from argparse import ArgumentParser
from pickle import load
from matplotlib import pyplot as plt

path.insert(1, ".")

parser = ArgumentParser()
parser.add_argument("checkpoint", type=str)
parser.add_argument("x", type=int, default=0)
parser.add_argument("y", type=int, default=1, nargs="+")

args = parser.parse_args()

with open(args.checkpoint, "rb") as f:
    chkpt = load(f)
    ts = chkpt["timeseries"]

fig = plt.figure(figsize=[8, 6])
ax1 = fig.add_subplot(111)

for i in args.y:
    x = [t[args.x] for t in ts]
    y = [t[i] for t in ts]
    ax1.plot(x, y, label=f"{chkpt['diagnostics'][i]}")
    ax1.set_xlabel(f"{chkpt['diagnostics'][args.x]}")

fig.legend()
plt.show()
