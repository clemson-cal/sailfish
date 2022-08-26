#!/usr/bin/env python3

import numpy as np
import pickle
import sys


def scale_by_n(infile, n):
    with open(infile, "rb") as inf:
        chk = pickle.load(inf)
    chk["solution"] = chk["solution"].repeat(n, axis=0).repeat(n, axis=1)
    chk["driver"] = chk["driver"]._replace(resolution=chk["driver"].resolution * n)
    chk["mesh"] = chk["mesh"]._replace(ni=chk["mesh"].ni * n, nj=chk["mesh"].nj * n)
    outfile = infile[:-3] + "_upscaled.pk"
    with open(outfile, "wb") as outf:
        pickle.dump(chk, outf)
    print(f"Written {infile} scaled by {n} to {outfile}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("No files provided, use 'python3 upsample.py file1 file2 ...'")
    for file in sys.argv[1:]:
        scale_by_n(file, 2)
