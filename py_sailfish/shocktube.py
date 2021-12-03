from sailfish.driver import run
import matplotlib.pyplot as plt

state = run("shocktube", end_time=0.2, resolution=1000)
rho = state["primitive"][:, 0]
plt.plot(rho)
plt.show()
