
import os
import sys

folder = "/home/gabriele/cifar10"
if not os.path.abspath(os.curdir) == folder: sys.exit()

CONTENT = \
f"""#!/bin/bash 
#SBATCH --job-name=SAVE_DIR
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=30GB 
#SBATCH --time=48:00:00 
#SBATCH --output {folder}/out_job/out_SAVE_DIR.txt 
#SBATCH --error {folder}/out_job/err_SAVE_DIR.txt 
ml purge 
ml Python/3.6.6-gomkl-2018b 
source /home/gabriele/iccv_tutto/myenv/bin/activate 
python {folder}/train.py --save_dir SAVE_DIR PARAMS
"""

for batch_size in range(4, 13):
    batch_size = f"{2 ** batch_size:04d}"
    save_dir = f"ab0_{batch_size}"
    filename = f"{folder}/jobs/{save_dir}.job"
    content = CONTENT.replace("SAVE_DIR", save_dir)\
                      .replace("PARAMS", f"--batch_size {batch_size}")
    with open(filename, "w") as file:
        _ = file.write(content)
    _ = os.system(f"sbatch {filename}")
    print(f"sbatch {filename}")

