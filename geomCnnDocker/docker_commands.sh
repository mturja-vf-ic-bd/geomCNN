#!/bin/bash

docker build -t geom_cnn_docker \
  --build-arg USER_ID=`id -u` \
  --build-arg GROUP_ID=`id -g` .

docker run -it -v /home/mturja/geomCNN:/home/mturja/geomCNN --gpus "device=0"  geom_cnn_docker:latest /bin/bash
cd /home/mturja/geomCNN

# random seeds
python3 -m src.training.EfficientNetTrainer --write_dir "/home/mturja/geomCNNlogsRandomSeeds" --backbone "mtl" --dropout 0.5 --batch_size 16 --learning_rate 0.0001 --in_channels 2 --num_classes 2 --gpus 1 --max_epochs 300 --n_folds 5 --exp_name "eacsf_V06_trial_1" &
python3 -m src.training.EfficientNetTrainer --write_dir "/home/mturja/geomCNNlogsRandomSeeds" --backbone "mtl" --dropout 0.5 --batch_size 16 --learning_rate 0.0001 --in_channels 2 --num_classes 2 --gpus 1 --max_epochs 300 --n_folds 5 --exp_name "eacsf_V06_trial_2" &
python3 -m src.training.EfficientNetTrainer --write_dir "/home/mturja/geomCNNlogsRandomSeeds" --backbone "mtl" --dropout 0.5 --batch_size 16 --learning_rate 0.0001 --in_channels 2 --num_classes 2 --gpus 1 --max_epochs 300 --n_folds 5 --exp_name "eacsf_V06_trial_3" &
python3 -m src.training.EfficientNetTrainer --write_dir "/home/mturja/geomCNNlogsRandomSeeds" --backbone "mtl" --dropout 0.5 --batch_size 16 --learning_rate 0.0001 --in_channels 2 --num_classes 2 --gpus 1 --max_epochs 300 --n_folds 5 --exp_name "eacsf_V06_trial_4" &
python3 -m src.training.EfficientNetTrainer --write_dir "/home/mturja/geomCNNlogsRandomSeeds" --backbone "mtl" --dropout 0.5 --batch_size 16 --learning_rate 0.0001 --in_channels 2 --num_classes 2 --gpus 1 --max_epochs 300 --n_folds 5 --exp_name "eacsf_V06_trial_5" &
python3 -m src.training.EfficientNetTrainer --write_dir "/home/mturja/geomCNNlogsRandomSeeds" --backbone "mtl" --dropout 0.5 --batch_size 16 --learning_rate 0.0001 --in_channels 2 --num_classes 2 --gpus 1 --max_epochs 300 --n_folds 5 --exp_name "eacsf_V06_trial_6" &
python3 -m src.training.EfficientNetTrainer --write_dir "/home/mturja/geomCNNlogsRandomSeeds" --backbone "mtl" --dropout 0.5 --batch_size 16 --learning_rate 0.0001 --in_channels 2 --num_classes 2 --gpus 1 --max_epochs 300 --n_folds 5 --exp_name "eacsf_V06_trial_7" &
python3 -m src.training.EfficientNetTrainer --write_dir "/home/mturja/geomCNNlogs-mtl-6"  --batch_size 500 --backbone "mtl"  --learning_rate 0.001 --dropout 0.1 --in_channels 6 --num_classes 2 --gpus 1 --max_epochs 500 --n_folds 3 &


# with more mlp layers
python3 -m src.training.EfficientNetTrainer --write_dir "/home/mturja/geomCNNlogs_skip" --backbone "mtl" --dropout 0.5 --batch_size 16 --learning_rate 0.0001 --in_channels 2 --num_classes 2 --gpus 1 --max_epochs 300 --n_folds 5 --exp_name "eacsf_V06_2layer_trial_1" &
python3 -m src.training.EfficientNetTrainer --write_dir "/home/mturja/geomCNNlogs_skip" --backbone "mtl" --dropout 0.5 --batch_size 16 --learning_rate 0.0001 --in_channels 2 --num_classes 2 --gpus 1 --max_epochs 300 --n_folds 5 --exp_name "eacsf_V06_2layer_trial_2" &
python3 -m src.training.EfficientNetTrainer --write_dir "/home/mturja/geomCNNlogs_skip" --backbone "mtl" --dropout 0.5 --batch_size 16 --learning_rate 0.0001 --in_channels 2 --num_classes 2 --gpus 1 --max_epochs 300 --n_folds 5 --exp_name "eacsf_V06_2layer_trial_3" &
python3 -m src.training.EfficientNetTrainer --write_dir "/home/mturja/geomCNNlogs_skip" --backbone "mtl" --dropout 0.5 --batch_size 16 --learning_rate 0.0001 --in_channels 2 --num_classes 2 --gpus 1 --max_epochs 300 --n_folds 5 --exp_name "eacsf_V06_2layer_trial_4" &


ssh -N -f -L localhost:20010:localhost:20010 mturja@theia.ia.unc.edu
ssh -N -f -L localhost:20010:localhost:20010 mturja@janus.ia.unc.edu
