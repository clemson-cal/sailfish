import numpy as np
import pickle
import argparse

# ------------------------------------------------------------------------------
# Current data format:
# dict(quantity="time"),
# dict(quantity="mdot", which_mass="1", gravity=True),
# dict(quantity="mdot", which_mass="1", accretion=True),
# dict(quantity="mdot", which_mass="2", gravity=True),
# dict(quantity="mdot", which_mass="2", accretion=True),
# dict(quantity="torque", which_mass="both", gravity=True),
# dict(quantity="torque", which_mass="both", accretion=True),
# dict(quantity="power", which_mass="both", gravity=True),
# dict(quantity="power", which_mass="both", accretion=True),
# ------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("filenames", nargs="+")
args = parser.parse_args()

for filename in args.filenames:
    chkpt = pickle.load(open(filename, "rb"))
    ts = np.array(chkpt["timeseries"])

    t = np.array([s[0].real for s in ts])
    mdot1_g = np.array([-s[1].real for s in ts])
    mdot1_a = np.array([-s[2].real for s in ts])
    mdot2_g = np.array([-s[3].real for s in ts])
    mdot2_a = np.array([-s[4].real for s in ts])
    torque_g = np.array([-s[5].real for s in ts])
    torque_a = np.array([-s[6].real for s in ts])
    power_g = np.array([s[7].real for s in ts])
    power_a = np.array([s[8].real for s in ts])

    for c1, c2, c3, c4, c5, c6, c7, c8, c9 in zip(
            t, mdot1_g, mdot1_a, mdot2_g, mdot2_a,
            torque_g, torque_a, power_g, power_a
    ):
        print(
            f"{c1:+6.5e} {c2:+6.5e} {c3:+6.5e} {c4:+6.5e} "
            f"{c5:+6.5e} {c6:+6.5e} {c7:+6.5e} {c8:+6.5e} {c9:+6.5e} "
        )
