#!/bin/bash
#
#SBATCH --job-name=MLP_RL
#SBATCH --output=/vinserver_user/21thinh.dd/PacMan_AI/scipts/MLP_train.out
#
#SBATCH --ntasks=1
#
sbcast -f /vinserver_user/21thinh.dd/PacMan_AI/scipts/run2.sh run2.sh
srun sh run.sh