#!/usr/bin/env python3

import argparse
import pickle


def main(args):
    import matplotlib.pyplot as plt

    for checkpoint in args.checkpoints:
        with open(checkpoint, "rb") as f:
            chkpt = pickle.load(f)

        try:
            rho = chkpt["primitive"][:, 0]
        except TypeError:
            rho = chkpt["solution"][:, 0]

        plt.plot(rho)
    plt.show()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("checkpoints", type=str, nargs="+")
    main(args.parse_args())
