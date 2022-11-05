
logdir = "/home/mturja/geomCNN_unet"
n_trials = 20
with open("run.sh", "w") as f:
    f.writelines("#!/bin/bash \n\n")
    for i in range(n_trials):
        f.writelines(f"python3 -m src.training.UNetTrainer --write_dir {logdir} --batch_size 16 --learning_rate 0.0001 --in_channels 4 --gpus 1 --max_epochs 200 --n_folds 5 --exp_name trial_{i} &\n")