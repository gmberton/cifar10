
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

for backbone in ["alexnet", "resnet18"]:
    for batch_size in [8, 16, 32, 64]:
        for lr in [0.01, 0.001, 0.0001]:
            save_dir = f"ab1_bb{backbone}_bs{batch_size}_lr{lr}"
            filename = f"{folder}/jobs/{save_dir}.job"
            content = CONTENT.replace("SAVE_DIR", save_dir)\
                              .replace("PARAMS", f"--backbone {backbone} --batch_size {batch_size} --lr {lr}")
            with open(filename, "w") as file:
                _ = file.write(content)
            _ = os.system(f"sbatch {filename}")
            print(f"sbatch {filename}")

