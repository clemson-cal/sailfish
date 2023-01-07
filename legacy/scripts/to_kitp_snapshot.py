import pickle
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("filenames", nargs="+")
args = parser.parse_args()

for filename in args.filenames:
    outfile = (
        filename.replace("-", "_")
        .replace("/", "_")
        .replace("chkpt.", "orbit_")
        .replace(".pk", ".bin")
    )
    print(f"{filename} -> {outfile}")

    # continue

    chkpt = pickle.load(open(filename, "rb"))
    mesh = chkpt["mesh"]
    solution = chkpt["solution"]

    x = [mesh.cell_coordinates(i, 0)[0] for i in range(mesh.shape[0])]
    y = [mesh.cell_coordinates(0, j)[1] for j in range(mesh.shape[1])]
    X, Y = np.meshgrid(x, y)

    data = np.zeros(mesh.shape + (5,))
    data[..., 0] = X.T
    data[..., 1] = Y.T
    data[..., 2] = solution[..., 0]
    data[..., 3] = solution[..., 1]
    data[..., 4] = solution[..., 2]
    np.array(data).tofile(outfile)
