import numpy as np
import time

filename = "workstation/array/G 1 results/mean velocities/1.25 1.0.txt"

vx = np.loadtxt(filename, dtype=float)

print(vx)