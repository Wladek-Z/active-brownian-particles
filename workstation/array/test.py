from pathlib import Path
import numpy as np
import os

path1 = 'G 1 results/MSD o0/0.25 0.5.txt'

with open(path1, 'rb') as f:
    try:  # catch OSError in case of a one line file 
        f.seek(-2, os.SEEK_END)
        while f.read(1) != b'\n':
            f.seek(-2, os.SEEK_CUR)
    except OSError:
        f.seek(0)
    last_line = f.readline().decode()

x, y = last_line.split(' ')

print(f"{x} {y}")
