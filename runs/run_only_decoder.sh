#!/bin/bash

export DISPLAY=:99.0
export PYVISTA_OFF_SCREEN=true
export PYVISTA_USE_IPYVTK=true
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
sleep 3

# N_CHANNELS="16 32 64 128"
N_CHANNELS="16 32 64 64"
# N_CHANNELS="128 128 128 128"

python main_decoder.py \
  -c config_files/config_folded_c_and_s.yaml \
  --n_channels_enc $N_CHANNELS \
  --latent_dim_c 32 \
  --latent_dim_s 64 \
  --w_kl 0 \
  --batch_size 512 \
  --z_aggr_function "DFT" \
  --learning_rate 0.00001 \
  --dataset.N_subjects 320 \
  --dataset.N_timeframes 4 \
  --dataset.amplitude_static_max 0.2 \
  --dataset.amplitude_dynamic_max 0.1 \
  --dataset.freq_max 2 \
  --dataset.l_max 1 \
  $@
  # --gpus ${GPU_DEVICE:-0} \

