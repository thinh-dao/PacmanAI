#!/bin/bash

# you have to cd to your workdir first
cd /vinserver_user/21thinh.dd/PacMan_AI

# another thing, you have to specify the full path of your python (you can determine this by running `which python` in the sandbox container
/home/admin/miniconda3/envs/duyna/bin/python -u train.py electricity forecast_multivar --loader forecast_csv --repr-dims 320 --max-threads 8 --seed 42 --eval