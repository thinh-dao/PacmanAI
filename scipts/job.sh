#!/bin/bash
#
#SBATCH --job-name=CNN_RL
#SBATCH --output=/vinserver_user/21thinh.dd/PacMan_AI/scipts/CNN_train.out
#
#SBATCH --ntasks=1
#
sbcast -f /vinserver_user/21thinh.dd/PacMan_AI/scipts/run.sh run.sh
srun sh run.sh