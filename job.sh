#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --output=/vinserver_user/duy.na184249/TS_Foundation_Model/CoInception/result.out
#
#SBATCH --ntasks=1
#
sbcast -f /vinserver_user/21thinh.dd/PacMan_AI/deepQLearningAgents.py/run.sh run.sh
srun sh run.sh