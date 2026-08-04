import numpy as np
from pathlib import Path

folder = Path("G 1/4.00 7.0")
files = sorted(folder.glob("*.txt"))
print(files[:20])
