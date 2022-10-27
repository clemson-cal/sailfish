import numpy as np
import pickle
import argparse

# dict(quantity="time"),
# dict(quantity="mdot", which_mass=1, accretion=True),
# dict(quantity="mdot", which_mass=2, accretion=True),
# dict(quantity="torque", which_mass="both", gravity=True),
# dict(
#     quantity="torque",
#     which_mass="both",
#     gravity=True,
#     radial_cut=(1.0, self.domain_radius),
# ),
# dict(quantity="sigma_m1"),
# dict(quantity="eccentricity_vector", radial_cut=(1.0, 6.0)),

parser = argparse.ArgumentParser()
parser.add_argument("filenames", nargs="+")
args = parser.parse_args()

for filename in args.filenames:
    chkpt = pickle.load(open(filename, "rb"))
    ts = np.array(chkpt["timeseries"])

    t = np.array([s[0].real for s in ts])
    mdot1 = np.array([-s[1].real for s in ts])
    mdot2 = np.array([-s[2].real for s in ts])
    torque_a = np.array([-s[3].real for s in ts])
    torque_b = np.array([-s[4].real for s in ts])
    psir = np.array([s[5].real for s in ts])
    psii = np.array([s[5].imag for s in ts])
    ex = np.array([s[6].real for s in ts])
    ey = np.array([s[6].imag for s in ts])

    for c1, c2, c3, c4, c5, c6, c7, c8, c9 in zip(
        t, torque_a, torque_b, psir, psii, ex, ey, mdot1, mdot2
    ):
        print(
            f"{c1:+6.5e} {c2:+6.5e} {c3:+6.5e} {c4:+6.5e} {c5:+6.5e} {c6:+6.5e} {c7:+6.5e} {c8:+6.5e} {c9:+6.5e}"
        )
