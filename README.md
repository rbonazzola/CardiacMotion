# Cardiac Motion's Representation Learning
**Author**: [Rodrigo Bonazzola](https://www.github.com/rbonazzola)

This repository contains code to perform representation learning on populations of CMR-derived cardiac meshes representing the movement of the heart across the cardiac cycle.

## Cloning this repository
To clone this repository, run:

```
git clone <REPO_LINK>
cd <REPO_FOLDER>
git submodule update --init --recursive
```

The last line will clone the Git submodules that this repository depends on, which is `VTKHelpers`.


## Scheme of the network

![SpatioTemporal_network](https://user-images.githubusercontent.com/11581216/167436436-15521711-8a8e-43a8-b6a5-564ba25e8232.png)


## Running the training script
Example command to perform network training and evaluation:

```
N_CHANNELS="128 128 128 128"

python main.py \
  -c config_files/config_folded_c_and_s.yaml \
  --n_channels_enc $N_CHANNELS \
  --n_channels_dec_c $N_CHANNELS \
  --n_channels_dec_s $N_CHANNELS \
  --latent_dim_c 32 \
  --latent_dim_s 64 \
  --w_s 1 \
  --w_kl 0 \
  --batch_size 512 \
  --z_aggr_function "DFT" \
  --learning_rate 0.00001 \
  --dataset.N_subjects 5120 \
  --dataset.N_timeframes 4 \
  --dataset.amplitude_static_max 0.2 \
  --dataset.amplitude_dynamic_max 0.1 \
  --dataset.freq_max 2 \
  --dataset.l_max 1 \
  --gpus ${GPU_DEVICE:-0}
```
