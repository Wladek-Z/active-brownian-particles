from pathlib import Path
import numpy as np

path1 = Path(__file__).with_name("Ps_params.txt")
path2 = Path(__file__).with_name("Pf_params.txt")

Ps_list = np.loadtxt(path1)
Pf_list = np.loadtxt(path2)

for Ps in set(Ps_list):
    print(f"{2*Ps - 1}\n")