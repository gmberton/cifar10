
from glob import glob
import numpy as np

import cl_utils as c

exps = sorted(glob("logs/*"))

S = """| so |   0  |   1  |   2  |   3  |   4  |   5  |   6  |   7  |   8  |   9  |  avg  |
| sw |      |      |      |      |      |      |      |      |      |      |       |
|----|------|------|------|------|------|------|------|------|------|------|-------|
"""

# for o in range(10):
#     print("    |", end="")

mat = np.zeros([10, 10])

for w in range(10):
    S += f"|  {w} |"
    for o in range(10):
        log = f"logs/sw_{w:02d}__so_{o:02d}/info.log"
        lines = c.readlines(log)
        # acc = c.get_matches("best_val_accuracy: (\d\d\.\d)", lines)[0]
        acc = c.get_matches("test_accuracy: (\d\d\.\d)", lines)[0]
        S += f" {acc} |"
        mat[w, o] = acc
    S += f" {mat[w, :].mean():.2f} |\n"

S += "| avg|"
for o in range(10):
    S += f" {mat[:, o].mean():.2f}|"

print(S)

