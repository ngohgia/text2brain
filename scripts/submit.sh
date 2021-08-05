#!/bin/bash

sbatch -p sablab --mem=28G --gres=gpu:1 --job-name=$1 -e ./job_err/%j-$1.err -o ./job_outputs/%j-$1.out $2
