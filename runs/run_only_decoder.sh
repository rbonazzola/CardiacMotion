#!/bin/bash

export DISPLAY=:99.0
export PYVISTA_OFF_SCREEN=true
export PYVISTA_USE_IPYVTK=true
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
sleep 3

# N_CHANNELS="16 32 64 128"
N_CHANNELS="16 32 64 64"
# N_CHANNELS="128 128 128 128"

((nvidia-smi &> /dev/null) && export DEVICE=${GPU_DEVICE:-0}) || export DEVICE="cpu"

python main.py \
  -c config_files/config_folded_c_and_s.yaml \
  --only_decoder \
  --n_channels_enc $N_CHANNELS \
  --n_channels_dec_c $N_CHANNELS \
  --n_channels_dec_s $N_CHANNELS \
  --w_kl 0 \
  --batch_size 256 \
  --z_aggr_function "DFT" \
  --learning_rate 0.0001 \
  --dataset.N_subjects 5120 \
  --dataset.N_timeframes 50 \
  --dataset.mesh_resolution 6 \
  --dataset.amplitude_static_max 0.2 \
  --dataset.amplitude_dynamic_max 0.1 \
  --dataset.freq_max 2 \
  --dataset.l_max 2 \
  --gpus ${GPU_DEVICE:-0} \
  $@

