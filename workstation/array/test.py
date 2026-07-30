from pathlib import Path
import numpy as np

path = Path(__file__).with_name("timechain10000000.txt")
tc = np.loadtxt(path)
tc2 = np.zeros_like(tc)
np.savetxt('testing.txt', np.column_stack((tc, tc2)), header='Time MSD')
