#!/bin/bash

python main.py \
  -c config/config_folded_c_and_s.yaml \
  --latent_dim_c 16 --latent_dim_s 32 \
  --w_s 0 \
  --w_kl 0 \
  --learning_rate 0.00001 \
  --gpus 1 \
  $1

