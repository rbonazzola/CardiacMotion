### Contents of this folder
The files in this folder implement the different components of the network. Finally, one file wraps the model with a PyTorch Lightning module.

| ID | Script | Description | Depends on  |
| ------ | ------ | ----------- | ------ |
| 1 | [`TemporalAggregators.py`](./TemporalAggregators.py) | `nn.Module`'s that implement temporal aggregation of latent vectors | - |
| 2 | [`PhaseModule.py`](./PhaseModule.py) | `nn.Module`'s that embed cardiac phase information into tensors | - |
| 3 | [`layers.py`](./layers.py) | Definition of the mesh-convolutional layers and downsampling/upsampling layers | - |
| 4 | [`model.py`](./model.py) | Full PyTorch model for the network | 1, 2, 3 |
| 5 | [`ComaLightningModule.py`](./ComaLightningModule.py) | Pytorch Lightning module | 4 | 
