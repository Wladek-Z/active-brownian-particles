import numpy as np

filename = "workstation/array/G 1 results/MSD ind/4.00 7.0 n20.npz"

data = np.load(filename)
t = data['time']
msds = data['MSD']


array = np.empty([len(t), len(msds) + 1])
array[:, 0] = t
string = "Time "

for i in range(1, len(msds) + 1):
    string += f"MSD{i} "
    array[:, i] = msds[i - 1]



np.savetxt("test_4.00_7.0_n20.txt", array, header=string, fmt='%.3f')