#!/bin/bash

set -euo pipefail

export DATA_DIR=./Synthetic_NeRF

python3 eval.py \
    --root_dir $DATA_DIR/Lego \
    --exp_name Lego \
    --batch_size 2048 --lr 1e-2 --gui
