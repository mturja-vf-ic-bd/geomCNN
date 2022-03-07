#!/bin/bash

docker build -t geomCnnDocker \
  --build-arg USER_ID=`id -u` \
  --build-arg GROUP_ID=`id -g` .

docker run -it -v /home/mturja:/home/mturja --gpus "device=1"  geom_cnn_docker:latest /bin/bash
cd /home/mturja/geomCNN
python3 -m src.training.EfficientNetTrainer --write_dir "/home/mturja/geomCNNlogs" --batch_size 64 --learning_rate 0.0001 --in_channels 2 --num_classes 2 --gpus 1 --max_epochs 500 --n_folds 5