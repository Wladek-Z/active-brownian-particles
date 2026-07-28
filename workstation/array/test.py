from pathlib import Path
import numpy as np

path = Path(__file__).with_name("timechain10000000.txt")
tc = np.loadtxt(path)
indices = np.arange(1, len(tc), 17)
print(np.arange(1, len(tc)-1)[np.diff((tc[1:] - tc[:-1]) == 1)][::2] + 1)
