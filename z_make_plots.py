
from glob import glob
import numpy as np

import cl_utils as c

exps = sorted(glob("logs/*"))

NUM_EXP = 100
FIXED_PARAM = "so"
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
    acc_val = c.get_matches("best_val_accuracy: (\d\d\.\d\d)", lines)[0]
    acc_test = c.get_matches("test_accuracy: (\d\d\.\d\d)", lines)[0]
    line_val += f" {acc_val}|"
    line_test += f" {acc_test}|"
    array_val[val] = acc_val
    array_test[val] = acc_test

S += "\n" + line_val + "\n" + line_test
print(S)


from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(array_val.reshape(-1, 1), array_test)
preds = regr.predict(array_val.reshape(-1, 1))


import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.scatter(array_val, array_test)
plt.plot(array_val, preds, color="blue", linewidth=3)
plt.xlim(77.5, 80)
plt.ylim(77.5, 80)
plt.gca().set_aspect('equal')
plt.grid()
plt.savefig(f"fixed_{FIXED_PARAM}.png", bbox_inches="tight", dpi=300)

