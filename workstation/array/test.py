from pathlib import Path
import numpy as np
import os


filename = '0.25 0.5.txt'

Ps, string = filename.split(' ')
Pf = string.split('.txt')[0]

print(f"Ps = {Ps}, Pf = {Pf}")
