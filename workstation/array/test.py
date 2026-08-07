from pathlib import Path
import numpy as np
import os


filename = 'tester.txt'

open(filename, 'w')

with open(filename, 'a') as f:
    f.write("goodbye")
