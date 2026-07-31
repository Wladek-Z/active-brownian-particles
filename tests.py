import numpy as np
import time

theta = np.pi / 3
e = np.array([np.cos(theta), np.sin(theta)])

start = time.time()
x = 0.5 * np.cos(2 * theta)
end = time.time()
print(f"Answer = {x}, Time = {end - start} s\n")

start = time.time()
y = 0.5 * (e[0]**2 - e[1]**2)
end = time.time()
print(f"Answer = {y}, Time = {end - start} s\n")

