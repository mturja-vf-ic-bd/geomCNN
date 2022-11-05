
logdir = "/home/mturja/geomCNNlogs_2layer_trials_V06_V12"
n_trials = 20
with open("run.sh", "w") as f:
    f.writelines("#!/bin/bash \n\n")
    for i in range(n_trials):
        f.writelines(f"python3 -m src.training.EfficientNetTrainer --write_dir {logdir} --backbone mtl --dropout 0.2 --batch_size 16 --learning_rate 0.0001 --in_channels 4 --num_classes 2 --gpus 1 --max_epochs 200 --n_folds 5 --exp_name trial_{i} &\n")