#!/bin/bash

export DISPLAY=:99.0
export PYVISTA_OFF_SCREEN=true
export PYVISTA_USE_IPYVTK=true
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
sleep 3

N_CHANNELS="16 32 64 128"
N_CHANNELS="64 64 128 128"

python main.py \
  -c config_files/config_folded_c_and_s.yaml \
  --n_channels_enc $N_CHANNELS \
  --n_channels_dec_c $N_CHANNELS \
  --n_channels_dec_s $N_CHANNELS \
  --latent_dim_c 16 \
  --latent_dim_s 32 \
  --w_s 0 \
  --w_kl 0 \
  --learning_rate 0.00001 \
  --gpus ${GPU_DEVICE:-0} \
  $@

