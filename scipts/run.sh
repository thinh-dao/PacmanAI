#!/bin/bash

# you have to cd to your workdir first
cd /vinserver_user/21thinh.dd/PacMan_AI

# another thing, you have to specify the full path of your python (you can determine this by running `which python` in the sandbox container
/home/admin/miniconda3/envs/21thinh.dd/bin/python -u /vinserver_user/21thinh.dd/PacMan_AI/pacman.py -p PacmanCNNQAgent -l mediumClassic -n 10000 -x 10000 


