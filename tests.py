import numpy as np

x = np.linspace(0.25, 2, 8)
print(x[x == 0.25])
print(x[~(x == 0.25)])
