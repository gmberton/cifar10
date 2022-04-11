
from glob import glob
import numpy as np

import cl_utils as c

exps = sorted(glob("logs/*"))

NUM_EXP = 100
FIXED_PARAM = "sw"
OTHER_PARAM = {"so": "sw", "sw": "so"}[FIXED_PARAM]

S = f"|    | {OTHER_PARAM} |"
for i in range(NUM_EXP):
    S += f"  {i:>2}  |"

S += "\n|---|-----|"
for i in range(NUM_EXP):
    S += "------|"

line_val = f"| val|{FIXED_PARAM}=0|"
line_test = f"|test|{FIXED_PARAM}=0|"

array_val = np.zeros([NUM_EXP])
array_test = np.zeros([NUM_EXP])

for val in range(NUM_EXP):
    if FIXED_PARAM == "sw":
        log = f"logs/sw_00__so_{val:02d}/info.log"
    else:
        log = f"logs/sw_{val:02d}__so_00/info.log"
    lines = c.readlines(log)
    acc_val = c.get_matches("best_val_accuracy: (\d\d\.\d)", lines)[0]
    acc_test = c.get_matches("test_accuracy: (\d\d\.\d)", lines)[0]
    line_val += f" {acc_val} |"
    line_test += f" {acc_test} |"
    array_val[val] = acc_val
    array_test[val] = acc_test

S += "\n" + line_val + "\n" + line_test
print(S)

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.scatter(array_val, array_test)
plt.savefig(f"fixed_{FIXED_PARAM}.png", bbox_inches="tight", dpi=300)

