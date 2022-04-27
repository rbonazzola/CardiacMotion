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


## Training

```
python main.py -c config/config_folded_c_and_s.yaml \
  --latent_dim_c 16 --latent_dim_s 32 \
  --w_s 0 --w_kl 0 \ 
  --learning_rate 0.00001 \
  --gpus 1 
```