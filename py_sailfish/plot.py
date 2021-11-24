import argparse
import pickle


def main(args):
    import matplotlib.pyplot as plt

    with open(args.checkpoint, "rb") as f:
        chkpt = pickle.load(f)

    rho = chkpt["primitive"][:, 0]
    plt.plot(rho)
    plt.show()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("checkpoint", type=str)
    main(args.parse_args())
