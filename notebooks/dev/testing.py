# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Coma
#     language: python
#     name: coma
# ---

# %% [markdown]
# # Development

# %%
import sys
sys.path.append("utils/VTKHelpers/")
from CardiacMesh import Cardiac3DMesh, Cardiac4DMesh, CardiacMeshPopulation
from models import layers

import yaml
from pprint import pprint
from argparse import Namespace
import logging

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import pytorch_lightning as pl

import ipywidgets as widgets
from IPython.display import display, HTML

# %%
javascript_functions = {False: "hide()", True: "show()"}
button_descriptions  = {False: "Show code", True: "Hide code"}


def toggle_code(state):

    """
    Toggles the JavaScript show()/hide() function on the div.input element.
    """

    output_string = "<script>$(\"div.input\").{}</script>"
    output_args   = (javascript_functions[state],)
    output        = output_string.format(*output_args)

    display(HTML(output))


def button_action(value):

    """
    Calls the toggle_code function and updates the button description.
    """

    state = value.new

    toggle_code(state)

    value.owner.description = button_descriptions[state]


state = True
toggle_code(state)

button = widgets.ToggleButton(state, description = button_descriptions[state])
button.observe(button_action, "value")
display(button)

# %% [markdown]
# # PyTorch Lightning DataModule

# %%
from utils.VTKHelpers.CardiacMesh import CardiacMeshPopulation, Cardiac3DMesh
from torch.utils.data import TensorDataset, DataLoader, random_split
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union


class CardiacMeshPopulationDataset(TensorDataset):
    
    '''
    PyTorch dataset wrapping the CardiacMeshPopulation class
    '''
    
    def __init__(
        self, 
        cardiac_population: Union[CardiacMeshPopulation, None]=None, 
        root_dir: Union[str,None]=None, 
        context=Namespace(logger=logging.getLogger())
    ):
        
        if cardiac_population is None and root_dir is None:            
            raise ValueError("Provide either cardiac_population or root_dir as argument")
        elif cardiac_population is not None and root_dir is not None:
            raise ValueError("Provide only one of cardiac_population or root_dir as argument")
        
        if root_dir is not None:
            cardiac_population = CardiacMeshPopulation(root_dir)
            
        self.ids = cardiac_population.ids
        self.data = torch.Tensor(cardiac_population.as_numpy_array())
        
        #TODO: check that this does not produce a copy of self.data (I think it does not)
        self._data_dict = { 
            self.ids[i]:self.data[i] for i, _ in enumerate(self.data) 
        }
        
    def __getitem__(self, id):        
        return self._data_dict[self.ids[id]]
        
    def __len__(self):
        return len(self.ids)


class CardiacMeshPopulationDM(pl.LightningDataModule):    
    
    '''
    PyTorch datamodule wrapping the CardiacMeshPopulation class
    '''
    
    def __init__(self, data_dir: str = "path/to/dir", 
                 batch_size: int = 32,
                 split_lengths: List[int]=[7, 3, 3]):
        
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.split_lengths = split_lengths

    def setup(self, stage: Optional[str] = None):
        popu = CardiacMeshPopulationDataset(root_dir=self.data_dir)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(popu, self.split_lengths)        

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=1, num_workers=8)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=1, num_workers=8)



# %% [markdown]
# # PyTorch Lightning Module

# %%
pl.LightningModule.on_train_epoch_start


# %% slideshow={"slide_type": "-"}
class CoMA(pl.LightningModule):
    
    def __init__(self, model, params):
        super(CoMA, self).__init__()
        self.model = model
        self.params = params
                
    
    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(input, **kwargs)
    
    def on_train_epoch_start(self):
        self.model.set_mode("training")
    
    def training_step(self, batch, batch_idx):
        
        print("holaaa")
        out = self(batch)#, mode="training") #
        train_loss = loss_function(out, batch)
        
        #Why not just passing
        self.logger.experiment.log({
            key: val.item() for key, val in train_loss.items()
        })
    
    
        return losses

    
    def training_epoch_end(self, outputs):
        # Aggregate metrics from each batch    
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log({"train_loss":avg_loss})
        pass
    
    
    def on_validation_epoch_start(self):
        self.model.set_mode("testing")

        
    def validation_step(self, batch, batch_idx):                
        
        out = self(batch)#, mode="training") #
        train_loss = loss_function(out, batch)
        
        #Why not just passing train_loss here?
        self.logger.experiment.log({
            key: val.item() for key, val in train_loss.items()
        })        

    
    def validation_epoch_end(self):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log({"val_loss":avg_loss})
        pass
    
    
    # def test_step(self):
    
    def test_epoch_end(self):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log({"val_loss":avg_loss})
        pass
    
    
    #TODO: Select optimizer from menu (dict)
    def configure_optimizers(self):
        
        algorithm = config.optimizer.algorithm
        algorithm = torch.optim.__dict__[algorithm]
        parameters = vars(config.optimizer.parameters)
        optimizer = algorithm(self.model.parameters(), **parameters)      
        return optimizer    

# %% [markdown]
# ### Testing PL module

# %%
from utils.generics import recursive_namespace


# %%
def load_config(yaml_config_file):
    import yaml
    with open(yaml_config_file) as config:
        config = yaml.safe_load(config)    
        # I am using a namespace instead of a dictionary mainly because it enables auto-completion
        config = recursive_namespace(config)
    
    config.network_architecture.convolution.parameters.polynomial_degree = \
    [int(x) for x in config.network_architecture.convolution.parameters.polynomial_degree.split()]
    
    config.network_architecture.pooling.parameters.downsampling_factors = \
    [int(x) for x in config.network_architecture.pooling.parameters.downsampling_factors.split()]
    
    config.network_architecture.convolution.parameters.channels = \
    [int(x) for x in config.network_architecture.convolution.parameters.channels.split()]
    
    return config


# %%
config = load_config("config/config.yaml")

# %%
dm = CardiacMeshPopulationDM(config.root_folder, batch_size=2)

# %%
from utils import mesh_operations
from utils.helpers import *

# %%
import os

# %%
# tmesh = Cardiac3DMesh(os.path.join(os.getcwd(), config.network_architecture.pooling.parameters.template_mesh))

# %%
tmesh = Cardiac3DMesh(config.network_architecture.pooling.parameters.template_mesh)

M, A, D, U = mesh_operations.generate_transform_matrices(
    mesh=tmesh, 
    factors=config.network_architecture.pooling.parameters.downsampling_factors
)

A_t, D_t, U_t = ([scipy_to_torch_sparse(x) for x in X] for X in (A, D, U))
n_nodes = [len(m.v) for m in M]

# %%
# init model

coma_args = {
  "num_features": config.network_architecture.n_features,
  "n_layers": len(config.network_architecture.convolution.parameters.channels), # REDUNDANT
  "num_conv_filters": config.network_architecture.convolution.parameters.channels,
  "polygon_order": config.network_architecture.convolution.parameters.polynomial_degree,
  "latent_dim": config.network_architecture.latent_dim,
  "is_variational": config.loss.regularization_loss.weight != 0,
  "downsample_matrices": D_t,
  "upsample_matrices": U_t, 
  "adjacency_matrices": A_t,
  "n_nodes": n_nodes, 
  "mode": "testing"
}

from models.model import Coma
coma = Coma4D(**coma_args)
model = CoMA(coma4D, config)

# %%
# train
trainer = pl.Trainer()
trainer.fit(model, datamodule=dm)

# %%
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint

# %% [markdown]
# ## Load data

# %%
# import objgraph
# objgraph.show_refs(config, max_depth=2, )

# %%
popu = CardiacMesh.CardiacMeshPopulation(config.root_folder)
print(popu.as_numpy_array().shape)

# %%
exp_it = [(np.sin(2*np.pi*i/len(popu.time_frames)), np.cos(2*np.pi*i/len(popu.time_frames))) for i in range(len(popu.time_frames))]
exp_it = np.array(exp_it)
exp_it = np.expand_dims(exp_it, axis=(0,2))
popu_array = popu.as_numpy_array()

time_embedded_shape = [
  np.array([
    exp_it[0,i,0,0]*popu_array[:,i,:,:], 
    exp_it[0,i,0,1]*popu_array[:,i,:,:]
  ]) for i in range(exp_it.shape[1])
]

# %%
time_embedded_shape = np.array(time_embedded_shape)
time_embedded_shape = time_embedded_shape.reshape([time_embedded_shape.shape[i] for i in (2,0,3,4,1)])

# %%
time_embedded_shape.shape

# %% [markdown]
# ## Network definition

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
# ## Optimizer definition

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
# ## Network training

# %%
