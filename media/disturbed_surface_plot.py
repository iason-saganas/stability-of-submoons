import matplotlib.pyplot as plt
import numpy as np

def p2(x):
    return 1/2 * (3*x**2 - 1)

def surface_eq(psi, R):
    return R*(1+0.001*p2(np.cos(psi)))

psi = np.linspace(0, 2*np.pi, 100)
R = surface_eq(psi, 1)

x = R*np.cos(psi)
y = R*np.sin(psi)

fig, ax = plt.subplots()

ax.plot(x, y)


# Remove frame, ticks, and labels
ax.set_frame_on(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])


fig.savefig("plot.svg", format="svg", transparent=True, bbox_inches='tight', pad_inches=0)
plt.show()