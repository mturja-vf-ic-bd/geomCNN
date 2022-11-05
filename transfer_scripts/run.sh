#!/bin/bash 

python3 -m src.training.UNetTrainer --write_dir /home/mturja/geomCNN_unet --batch_size 4 --learning_rate 0.00001 --in_channels 4 --gpus 1 --max_epochs 200 --n_folds 5 --exp_name trial_0
python3 -m src.training.UNetTrainer --write_dir /home/mturja/geomCNN_unet --batch_size 16 --learning_rate 0.0001 --in_channels 4 --gpus 1 --max_epochs 200 --n_folds 5 --exp_name trial_1 &
python3 -m src.training.UNetTrainer --write_dir /home/mturja/geomCNN_unet --batch_size 16 --learning_rate 0.0001 --in_channels 4 --gpus 1 --max_epochs 200 --n_folds 5 --exp_name trial_2 &
python3 -m src.training.UNetTrainer --write_dir /home/mturja/geomCNN_unet --batch_size 16 --learning_rate 0.0001 --in_channels 4 --gpus 1 --max_epochs 200 --n_folds 5 --exp_name trial_3 &
python3 -m src.training.UNetTrainer --write_dir /home/mturja/geomCNN_unet --batch_size 16 --learning_rate 0.0001 --in_channels 4 --gpus 1 --max_epochs 200 --n_folds 5 --exp_name trial_4 &
python3 -m src.training.UNetTrainer --write_dir /home/mturja/geomCNN_unet --batch_size 16 --learning_rate 0.0001 --in_channels 4 --gpus 1 --max_epochs 200 --n_folds 5 --exp_name trial_5 &
python3 -m src.training.UNetTrainer --write_dir /home/mturja/geomCNN_unet --batch_size 16 --learning_rate 0.0001 --in_channels 4 --gpus 1 --max_epochs 200 --n_folds 5 --exp_name trial_6 &
python3 -m src.training.UNetTrainer --write_dir /home/mturja/geomCNN_unet --batch_size 16 --learning_rate 0.0001 --in_channels 4 --gpus 1 --max_epochs 200 --n_folds 5 --exp_name trial_7 &
python3 -m src.training.UNetTrainer --write_dir /home/mturja/geomCNN_unet --batch_size 16 --learning_rate 0.0001 --in_channels 4 --gpus 1 --max_epochs 200 --n_folds 5 --exp_name trial_8 &
python3 -m src.training.UNetTrainer --write_dir /home/mturja/geomCNN_unet --batch_size 16 --learning_rate 0.0001 --in_channels 4 --gpus 1 --max_epochs 200 --n_folds 5 --exp_name trial_9 &
python3 -m src.training.UNetTrainer --write_dir /home/mturja/geomCNN_unet --batch_size 16 --learning_rate 0.0001 --in_channels 4 --gpus 1 --max_epochs 200 --n_folds 5 --exp_name trial_10 &
python3 -m src.training.UNetTrainer --write_dir /home/mturja/geomCNN_unet --batch_size 16 --learning_rate 0.0001 --in_channels 4 --gpus 1 --max_epochs 200 --n_folds 5 --exp_name trial_11 &
python3 -m src.training.UNetTrainer --write_dir /home/mturja/geomCNN_unet --batch_size 16 --learning_rate 0.0001 --in_channels 4 --gpus 1 --max_epochs 200 --n_folds 5 --exp_name trial_12 &
python3 -m src.training.UNetTrainer --write_dir /home/mturja/geomCNN_unet --batch_size 16 --learning_rate 0.0001 --in_channels 4 --gpus 1 --max_epochs 200 --n_folds 5 --exp_name trial_13 &
python3 -m src.training.UNetTrainer --write_dir /home/mturja/geomCNN_unet --batch_size 16 --learning_rate 0.0001 --in_channels 4 --gpus 1 --max_epochs 200 --n_folds 5 --exp_name trial_14 &
python3 -m src.training.UNetTrainer --write_dir /home/mturja/geomCNN_unet --batch_size 16 --learning_rate 0.0001 --in_channels 4 --gpus 1 --max_epochs 200 --n_folds 5 --exp_name trial_15 &
python3 -m src.training.UNetTrainer --write_dir /home/mturja/geomCNN_unet --batch_size 16 --learning_rate 0.0001 --in_channels 4 --gpus 1 --max_epochs 200 --n_folds 5 --exp_name trial_16 &
python3 -m src.training.UNetTrainer --write_dir /home/mturja/geomCNN_unet --batch_size 16 --learning_rate 0.0001 --in_channels 4 --gpus 1 --max_epochs 200 --n_folds 5 --exp_name trial_17 &
python3 -m src.training.UNetTrainer --write_dir /home/mturja/geomCNN_unet --batch_size 16 --learning_rate 0.0001 --in_channels 4 --gpus 1 --max_epochs 200 --n_folds 5 --exp_name trial_18 &
python3 -m src.training.UNetTrainer --write_dir /home/mturja/geomCNN_unet --batch_size 16 --learning_rate 0.0001 --in_channels 4 --gpus 1 --max_epochs 200 --n_folds 5 --exp_name trial_19 &
