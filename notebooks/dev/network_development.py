# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os, sys
os.chdir("..")

# %%
sys.path.append(os.getcwd())

# %%
# from models.Model3D import Encoder3DMesh, Decoder3DMesh
# from .PhaseModule import PhaseTensor
# from .TemporalAggregators import Mean_Aggregator, DFT_Aggregator, FCN_Aggregator

# %%
from easydict import EasyDict
from models.layers import ChebConv_Coma, Pool
from typing import Sequence, Union, List
from copy import copy

import torch
from torch import nn
from torch.nn import ModuleList, ModuleDict
from torch.fft import rfft

from data.SyntheticDataModules import SyntheticMeshesDataset, SyntheticMeshesDM

import numpy as np
from copy import copy
from typing import Sequence, Union, List
from utils.helpers import get_coma_args

from IPython import embed # left there for debugging if needed


# %% [markdown]
# ____
# ## 3D models

# %%
#TODO: Implement common parent class for encoder and decoder (GraphConvStack?), to capture common behaviour.

################# FULL AUTOENCODER #################

class Autoencoder3DMesh(nn.Module):

    def __init__(self, enc_config, dec_config):

        super(Autoencoder3DMesh, self).__init__()

        self.encoder = Encoder3DMesh(**enc_config)
        self.decoder = Decoder3DMesh(**dec_config)

    def forward(self, x):

        mu, logvar = self.encoder(x)
        # Add sampling if is_variational == True and it's in training mode
        x_hat = self.decoder(mu)
        return x_hat

################# ENCODER #################

ENCODER_ARGS = [
    "num_features",
    "n_layers",
    "n_nodes",
    "num_conv_filters_enc",
    "cheb_polynomial_order",
    "latent_dim_content",
    "template",
    "is_variational",
    "phase_input",
    "downsample_matrices",
    "adjacency_matrices",
    "activation_layers"
]

class Encoder3DMesh(nn.Module):

    '''
    '''

    def __init__(self,
        phase_input: bool,
        num_conv_filters_enc: Sequence[int],
        num_features: int,
        cheb_polynomial_order: int,
        n_layers: int,
        n_nodes: int,
        is_variational: bool,
        latent_dim: int,
        template,
        adjacency_matrices: List[torch.Tensor],
        downsample_matrices: List[torch.Tensor],
        activation_layers="ReLU"):

        super(Encoder3DMesh, self).__init__()

        self.n_nodes = n_nodes
        self.phase_input = phase_input
        self.filters_enc = copy(num_conv_filters_enc)
        self.filters_enc.insert(0, num_features)
        self.K = cheb_polynomial_order

        self.matrices = {}
        A_edge_index, A_norm = self._build_adj_matrix(adjacency_matrices)

        self.matrices["A_edge_index"] = A_edge_index
        self.matrices["A_norm"] = A_norm
        self.matrices["downsample"] = downsample_matrices
                
        self._n_features_before_z = self.matrices["downsample"][-1].shape[0] * self.filters_enc[-1]
        self._is_variational = is_variational
        self.latent_dim = latent_dim

        self.activation_layers = [activation_layers] * n_layers if isinstance(activation_layers, str) else activation_layers
        self.layers = self._build_encoder()

        # Fully connected layers connecting the last pooling layer and the latent space layer.
        self.enc_lin_mu = torch.nn.Linear(self._n_features_before_z, self.latent_dim)

        if self._is_variational:
            self.enc_lin_var = torch.nn.Linear(self._n_features_before_z, self.latent_dim)

    def _build_encoder(self):

        cheb_conv_layers = self._build_cheb_conv_layers(self.filters_enc, self.K)
        pool_layers = self._build_pool_layers(self.matrices["downsample"])
        activation_layers = self._build_activation_layers(self.activation_layers)

        encoder = ModuleDict()

        for i in range(len(cheb_conv_layers)):
            layer = f"layer_{i}"
            encoder[layer] = ModuleDict()            
            encoder[layer]["graph_conv"] = cheb_conv_layers[i]
            encoder[layer]["pool"] = pool_layers[i]
            encoder[layer]["activation_function"] = activation_layers[i]

        return encoder

    def _build_pool_layers(self, downsample_matrices:Sequence[np.array]):

        '''
        downsample_matrices: list of matrices binary matrices
        '''

        pool_layers = ModuleList()
        for i in range(len(downsample_matrices)):
            pool_layers.append(Pool())
        return pool_layers


    def _build_activation_layers(self, activation_type:Union[str, Sequence[str]]):

        '''
        activation_type: string or list of strings containing the name of a valid activation function from torch.functional
        '''

        activation_layers = ModuleList()

        for i in range(len(activation_type)):
            if activation_type[i] is None:
                activ_fun = torch.nn.Identity()
            else: 
                activ_fun = getattr(torch.nn.modules.activation, activation_type[i])()
            activation_layers.append(activ_fun)

        return activation_layers


    def _build_cheb_conv_layers(self, n_filters, K):
        # Chebyshev convolutions (encoder)

        # TOFIX: this should be specified in the docs.
        if self.phase_input:
            n_filters[0] = 2 * n_filters[0]

        cheb_enc = torch.nn.ModuleList([ChebConv_Coma(n_filters[0], n_filters[1], K[0])])
        cheb_enc.extend([
            ChebConv_Coma(
                n_filters[i],
                n_filters[i+1],
                K[i]
            ) for i in range(1, len(n_filters)-1)
        ])
        return cheb_enc


    def _build_adj_matrix(self, adjacency_matrices):
        adj_edge_index, adj_norm = zip(*[
            ChebConv_Coma.norm(adjacency_matrices[i]._indices(), self.n_nodes[i])
            for i in range(len(self.n_nodes))
        ])
        return list(adj_edge_index), list(adj_norm)

    
    def concatenate_graph_features(self, x):
        # embed()
        x = x.reshape(x.shape[0], x.shape[1], -1)
        # x = x.reshape(x.shape[0], self._n_features_before_z)
        return x


    # perform a forward pass only through the convolutional stack (not the FCN layer)
    def forward_conv_stack(self, x, preserve_graph_structure=True):
        
        # a "layer" here is: a graph convolution + pooling operation + activation function
        for i, layer in enumerate(self.layers): 
            
            if self.matrices["downsample"][i].device != x.device:
                self.matrices["downsample"][i] = self.matrices["upsample"][i].to(x.device)
            if self.matrices["A_edge_index"][i].device != x.device:
                self.matrices["A_edge_index"][i] = self.matrices["A_edge_index"][i].to(x.device)
            if self.matrices["A_norm"][i].device != x.device:
                self.matrices["A_norm"][i] = self.matrices["A_norm"][i].to(x.device)
  
            x = self.layers[layer]["graph_conv"](x, self.matrices["A_edge_index"][i], self.matrices["A_norm"][i])
            try:
                x = self.layers[layer]["pool"](x, self.matrices["downsample"][i])
            except:
                embed()
            x = self.layers[layer]["activation_function"](x)
        
        if not preserve_graph_structure:
            x = self.concatenate_graph_features(x)
            
        return x
    
    
    def forward(self, x):

        x = self.forward_conv_stack(x)       
        mu = self.enc_lin_mu(x)
        log_var = self.enc_lin_var(x) if self._is_variational else None        
        return {"mu": mu, "log_var": log_var}

    
################# DECODER #################

DECODER_ARGS = [
    "num_features",
    "n_layers",
    "n_nodes",
    "num_conv_filters_dec",
    "cheb_polynomial_order",
    "latent_dim_content",
    "is_variational",
    "upsample_matrices",
    "adjacency_matrices",
    "activation_layers",
    "template"
]

# %% [markdown]
# ## 4D models

# %% [markdown]
# ### Temporal Aggregators

# %%
BATCH_DIMENSION = 0
TIME_DIMENSION = 1
NODE_DIMENSION = 2
FEATURE_DIMENSION = 3

COMMON_ARGS = [
    "num_features",
    "n_layers",
    "n_nodes",
    "cheb_polynomial_order",
    "is_variational",
    "adjacency_matrices",
    "activation_layers",
    "template",
]

ENCODER_ARGS = copy(COMMON_ARGS)
ENCODER_ARGS.extend([
  "phase_input",
  "downsample_matrices",
  "num_conv_filters_enc",
  "latent_dim_content",
  "latent_dim_style"
])

DECODER_C_ARGS = copy(COMMON_ARGS)
DECODER_C_ARGS.extend([
  "upsample_matrices",
  "num_conv_filters_dec_c",
  "latent_dim_content"
])

DECODER_S_ARGS = copy(COMMON_ARGS)
DECODER_S_ARGS.extend([
    "upsample_matrices",
    "num_conv_filters_dec_s",
    "latent_dim_content",
    "latent_dim_style",
    "n_timeframes"
])


# %%
def _steal_attributes_from_child(self, child: str, attributes: Union[List[str], str]):

    '''
       Make attributes from an object's child visible from the (parent) object's namespace
    '''

    child = getattr(self, child)

    if isinstance(attributes, str):
        attributes = [attributes]

    for attribute in attributes:
        setattr(self, attribute, getattr(child, attribute))
    return self


##########################################################################################

class EncoderTemporalSequence(nn.Module):

    def __init__(self, encoder3d, z_aggr_function, phase_embedding=None, n_timeframes=None):

        '''
        
        
        '''
        
        super(EncoderTemporalSequence, self).__init__()
        # encoder_config = copy(encoder_config)
        # encoder_config["latent_dim"] = encoder_config.pop("latent_dim_content") + encoder_config.pop("latent_dim_style")
        # 
        # self.latent_dim = encoder_config["latent_dim"]
        # Encoder3DMesh(**encoder_config)
        self.encoder_3d_mesh = encoder3d

        self = _steal_attributes_from_child(self, child="encoder_3d_mesh", attributes=["matrices"])

        # self.z_aggr_function = self._get_z_aggr_function(z_aggr_function, n_timeframes)
        self.z_aggr_function = z_aggr_function
        self.phase_embedding = phase_embedding


    def set_mode(self, mode: str):
        '''
        params:
          mode: "training" or "testing"
        '''
        self._mode = mode


    def encoder(self, x):
                
        self.n_timeframes = x.shape[1]

        # Iterate through time points
        # 
        
        # embed()
        
        #bottleneck_t = [ self.encoder_3d_mesh(x[:, i, :]) for i in range(self.n_timeframes) ]
        #mu = [ bottleneck["mu"] for bottleneck in bottleneck_t ]

        # If one element (and therefore all elements) are None, replace the whole thing with None
        #log_var = [ bottleneck["log_var"] for bottleneck in bottleneck_t ] if bottleneck_t[0]["log_var"] is not None else None

        #mu = torch.cat(mu).reshape(-1, self.n_timeframes, self.latent_dim)
        h = self.encoder_3d_mesh.forward_conv_stack(x, preserve_graph_structure=False)
        z = self.z_aggr_function(h)

        # if log_var is not None:
        #     log_var_t = torch.cat(log_var).reshape(-1, self.n_timeframes, self.latent_dim)
        #     log_var = self.z_aggr_function(log_var_t)

        bottleneck = {"mu": z, "log_var": None}
        return bottleneck


    def forward(self, x):
        return self.encoder(x)


# %% [markdown]
# ## Lightning module

# %%
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from PIL import Image
import imageio
import numpy as np

from IPython import embed # uncomment for debugging
# from models.Model4D import  EncoderTemporalSequence
# from data.synthetic.SyntheticMeshPopulation import SyntheticMeshPopulation

from image_helpers import *

losses_menu = {
  "l1": F.l1_loss,
  "mse": F.mse_loss
}

def mse(s1, s2=None):
    if s2 is None:
        s2 = torch.zeros_like(s1)
    return ((s1-s2)**2).sum(-1).mean(-1)

LOG_OPTIONS = {
    "on_epoch": True,
    "prog_bar": True,
    "logger": True
}

class TemporalEncoderLightning(pl.LightningModule):

    def __init__(self, model, params):

        '''
        :param model: provide the PyTorch model.
        :param params: a Namespace with additional parameters
        
        Example:
        
        
        '''

        super(TemporalEncoderLightning, self).__init__()
        self.model = model
        self.params = params

        self.rec_loss = self.get_rec_loss()


    def get_rec_loss(self):

        self.w_s = self.params.loss.reconstruction_s.weight
        self.w_kl = self.params.loss.regularization.weight
        return losses_menu[self.params.loss.reconstruction_c.type.lower()]


    def on_fit_start(self):

        #TODO: check of alternatives since .to(device) is not recommended
        #This is the most elegant way I found so far to transfer the tensors to the right device
        #(if this is run within __init__, I get self.device=="cpu" even when I use a GPU, so it doesn't work there)

        for matrix_type in ["downsample", "A_edge_index", "A_norm"]:
            for i, _ in enumerate(self.model.matrices["downsample"]):
                self.model.matrices[matrix_type][i] = self.model.matrices[matrix_type][i].to(self.device)


    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(input, **kwargs)


    def on_train_epoch_start(self):
        self.model.set_mode("training")


    def training_step(self, batch, batch_idx):

        # data, ids = batch
        # TODO: change dataset/datamodule accordingly
        s_t, z_c, z_s = batch["s_t"], batch["z_c"], batch["z_s"]

        z = z_c + z_s  # list concatenation
        z = torch.stack(z).transpose(0, 1).type_as(s_t)  # to get N_batches x latent_dim

        bottleneck = self(s_t)
        z_hat = bottleneck["mu"]
        recon_loss = self.rec_loss(z, z_hat)

        loss_dict = {
           "training_recon_loss": recon_loss,
           "loss": recon_loss
        }

        # https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#log-dict
        self.log_dict(loss_dict)
        return loss_dict


    def training_epoch_end(self, outputs):

        # Aggregate metrics from each batch

        avg_recon_loss = torch.stack([x["training_recon_loss"] for x in outputs]).mean()
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

        self.log_dict({
            "training_recon_loss": avg_recon_loss.detach(),
            "training_loss": avg_loss.detach()
          }, **LOG_OPTIONS          
        )

    def on_validation_epoch_start(self):
        self.model.set_mode("testing")


    def _shared_eval_step(self, batch, batch_idx, prefix: str):

        '''
        The validation and testing steps are similar, only the names of the logged quantities differ.
        The common part is performed here.
        '''

        s_t, z_c, z_s = batch["s_t"], batch["z_c"], batch["z_s"]

        z = z_c + z_s # list concatenation
        z = torch.stack(z).transpose(0,1).type_as(s_t) # to get N_batches x latent_dim

        bottleneck = self(s_t)
        z_hat = bottleneck["mu"]

        recon_loss = self.rec_loss(z, z_hat)
        loss = recon_loss

        # TODO: compute normalized metrics
        z_sqerr = (z - z_hat)**2
        z_sqnorm = z**2
        
        loss_dict = { 
            f"{prefix}recon_loss": recon_loss, 
            f"{prefix}loss": loss, 
            f"{prefix}z_sqerr": z_sqerr, 
            f"{prefix}z_sqnorm": z_sqnorm 
        }
        
        self.log_dict(loss_dict)
        
        return loss_dict


    def validation_step(self, batch, batch_idx):

        loss_dict = self._shared_eval_step(batch, batch_idx, prefix="val_")
                
        return loss_dict


    def validation_epoch_end(self, outputs):

        #TODO: iterate over keys of the elements of `outputs`

        avg_recon_loss = torch.stack([x["val_recon_loss"] for x in outputs]).mean()
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()        
        
        avg_z_sqerr = torch.stack([x["val_z_sqerr"] for x in outputs]).mean()
        avg_z_sqnorm = torch.stack([x["val_z_sqnorm"] for x in outputs]).mean()                
        avg_norm_z_err = avg_z_sqerr / avg_z_sqnorm

        self.log_dict({
            "val_recon_loss": avg_recon_loss.detach(), 
            "val_loss": avg_loss.detach(),
            "val_z_norm_err": avg_norm_z_err.detach()            
          }, 
          **LOG_OPTIONS          
        )


    def test_step(self, batch, batch_idx):

        loss_dict = self._shared_eval_step(batch, batch_idx, prefix="test_")        
        return loss_dict


    #TOFIX: DUPLICATED FROM ABOVE
    def test_epoch_end(self, outputs):

        avg_recon_loss = torch.stack([x["test_recon_loss"] for x in outputs]).mean()
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        
        avg_z_sqerr = torch.stack([x["test_z_sqerr"] for x in outputs]).mean()
        avg_z_sqnorm = torch.stack([x["test_z_sqnorm"] for x in outputs]).mean()                
        avg_norm_z_err = avg_z_sqerr / avg_z_sqnorm


        loss_dict = {
            "test_recon_loss": avg_recon_loss.detach(),
            "test_loss": avg_loss.detach(),
            "test_z_norm_err": avg_norm_z_err.detach()
        }

        self.log_dict(loss_dict)


    def configure_optimizers(self):

        algorithm = self.params.optimizer.algorithm
        algorithm = torch.optim.__dict__[algorithm]
        parameters = vars(self.params.optimizer.parameters)
        optimizer = algorithm(self.model.parameters(), **parameters)
        return optimizer


# %%
class Mean_Aggregator(nn.Module):
    
    '''
    '''
    
    def forward(self, x):
        return torch.Tensor.mean(x, axis=1)


class FCN_Aggregator(nn.Module):

    def __init__(self, features_in, features_out):
        
        '''
        '''
        
        super(FCN_Aggregator, self).__init__()
        self.fcn = torch.nn.Linear(features_in, features_out)

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        return self.fcn(x)


class DFT(nn.Module):
    
    '''
      Real discrete Fourier transform (DFT)    
    '''
    
    def forward(self, x):
        return rfft(x, dim=1)
    
    
class DFT_Aggregator(nn.Module):

    '''
      A DFT operator followed by a fully connected layer
      DFT: x [N, T, ..., F] -> [N, ..., n_comps * F]
      FCN:            
    '''

    def __init__(self, features_in, features_out):
        
        '''
        
        '''
        
        super(DFT_Aggregator, self).__init__()
        self.dft = DFT()
        self.fcn = torch.nn.Linear(features_in, features_out)
        
                
    def forward(self, x):

        x = self.dft(x)
        # Concatenate features in the frequency domain
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        x = torch.cat((x.real, x.imag), dim=-1)
        x = self.fcn(x)
        return x

    
class TemporalAggregator(nn.Module):
    
    def __init__(self, n_timeframes, n_spatial_features, n_hidden, latent_dim):
        
        '''
        Example:
        
            # h was the output of a previous computation
            # h = previous_module(x)
            n_spatial_f = h.shape[-1]
            
            # geometric mean
            n_hidden = int(np.sqrt(n_spatial_f * latent_dim))
            
            TAggr = TemporalAggregator(
              n_timeframes=20, # n_timeframes = config.dataset.parameters.T
              n_spatial_features=n_spatial_f,
              n_hidden=n_hidden,
              latent_dim=latent_dim
            )        
            
            z = TAggr(h)
            
        '''
        
        super(TemporalAggregator, self).__init__()
        
        self.z_taggr = DFT_Aggregator(
          features_in=(n_timeframes // 2 + 1) * 2 * n_spatial_features,
          features_out=n_hidden
        )
        
        self.sigmoid = nn.Sigmoid()
        self.fcn = nn.Linear(in_features=n_hidden, out_features=latent_dim)

        
    def forward(self, x):
        z = self.z_taggr(x)
        z = self.sigmoid(z)
        z = self.fcn(z)
        return z

    
    #def _get_z_aggr_function(self, z_aggr_function, n_timeframes=None):
    #
    #    if z_aggr_function == "mean":
    #        if phase_embedding is None:
    #            exit("The temporal aggregation cannot be the mean if phase information is not embedded into the input meshes.")
    #        z_aggr_function = Mean_Aggregator()
    #
    #    elif z_aggr_function.lower() in {"fcn", "fully_connected"}:
    #        self.n_timeframes = n_timeframes
    #        z_aggr_function = FCN_Aggregator(
    #            features_in=n_timeframes * self.latent_dim,
    #            features_out=(self.latent_dim)
    #        )
    #
    #    elif z_aggr_function.lower() in {"dft", "discrete_fourier_transform"}:
    #        self.n_timeframes = n_timeframes
    #        features_in = (n_timeframes // 2 + 1) * 2 * (self.latent_dim)
    #        features_out = (self.latent_dim)
    #        z_aggr_function = DFT_Aggregator(
    #            features_in=features_in,
    #            features_out=features_out
    #        )
    #
    #    return z_aggr_function            

# %% [markdown]
# ---
# # <center> --- Tests --- <center>

# %%
from config.load_config import load_yaml_config
config = load_yaml_config("config_files/config_folded_c_and_s.yaml")
config.dataset.parameters.N = 5120

mesh_ds = SyntheticMeshesDataset(config.dataset.parameters, config.dataset.preprocessing)
mesh_dm = SyntheticMeshesDM(mesh_ds)
mesh_dm.setup()

# %%
coma_args = get_coma_args(config, mesh_dm.dataset)

# %% [markdown]
# Let's define the 3D encoder:

# %%
x = EasyDict(next(iter(mesh_dm.train_dataloader())))

# %%
enc_params = {
    "phase_input" : False, 
    "num_conv_filters_enc" : [16, 16, 16, 16], 
    "num_features" : 3,
    "cheb_polynomial_order" : [6, 6, 6, 6],
    "n_layers" : 4,
    "n_nodes" : coma_args.n_nodes,
    "is_variational" : True,
    "latent_dim" : len(x.z_c) + len(x.z_s),
    "template": coma_args.template,
    "adjacency_matrices": coma_args.adjacency_matrices,
    "downsample_matrices": coma_args.downsample_matrices,
    "activation_layers": ["Sigmoid", "Sigmoid", "Sigmoid", None]
}

enc_params = EasyDict(enc_params)

encoder = Encoder3DMesh(**enc_params)

# %%
h = encoder.forward_conv_stack(x.s_t, preserve_graph_structure=False)

# %%
z_taggr = FCN_Aggregator(
    features_in=20*h.shape[-1],
    features_out= enc_params.latent_dim
)

# %%
h = encoder.forward_conv_stack(x.s_t, preserve_graph_structure=False)
z = z_taggr(h)

# %%
t_encoder = EncoderTemporalSequence(
    encoder3d=encoder,
    z_aggr_function=z_taggr
)

t_encoder_ptl = TemporalEncoderLightning(
  model=t_encoder,
  params=config
)

trainer = pl.Trainer(gpus=1, precision=16)

# %%
config.optimizer.parameters.lr = 3e-4

# %%
# trainer.fit(
#     model=t_encoder_ptl,
#     datamodule=mesh_dm
# )

# %%
# EPOCH = 999
# PATH = "model2.pt"
# 
# torch.save({
#     'epoch': EPOCH,
#     'model_state_dict': t_encoder_ptl.state_dict(),
#     'optimizer_state_dict': t_encoder_ptl.optimizers().state_dict(),
#     # 'loss': LOSS,
# }, PATH)

# %%
# chkpt = torch.load(PATH)
# model = 

# %%
z = t_encoder(x.s_t)

# %% [markdown]
# ## Decoder

# %%
import importlib
import models.Model3D
import models.Model4D
from models.Model3D import Decoder3DMesh
from models.Model4D import DECODER_C_ARGS, DECODER_S_ARGS
from models.Model4D import DecoderStyle, DecoderContent, DecoderTemporalSequence

# %%
# models.Model4D = importlib.reload(models.Model4D)
# models.Model3D = importlib.reload(models.Model3D)
# Decoder3DMesh = models.Model3D.Decoder3DMesh
# DecoderStyle, DecoderContent, DecoderTemporalSequence = models.Model4D.DecoderStyle, models.Model4D.DecoderContent, models.Model4D.DecoderTemporalSequence

# %%
coma_args.latent_dim_content = 9
coma_args.latent_dim_style = 36

# %%
decoder_config_c = { k:v for k,v in coma_args.items() if k in DECODER_C_ARGS }
decoder_config_s = { k:v for k,v in coma_args.items() if k in DECODER_S_ARGS }

# %%
decoder_content = DecoderContent(decoder_config_c)

# %%
decoder_content = DecoderContent(decoder_config_c)
decoder_style = DecoderStyle(decoder_config_s, phase_embedding_method="exp")
t_decoder = DecoderTemporalSequence(decoder_content, decoder_style)

# %% [markdown]
# ## Autoencoder

# %%
from models.Model4D import AutoencoderTemporalSequence

# %%
t_ae = AutoencoderTemporalSequence(
    encoder=t_encoder,
    decoder=t_decoder
)

# %%
decoder_style.decoder_3d.latent_dim

# %%
from lightning.ComaLightningModule import CoMA_Lightning

# %%
lit_module = CoMA_Lightning(
    model=t_ae, 
    loss_params=config.loss, 
    optimizer_params=config.optimizer,
    additional_params=config
)

# %%
t_ae(x.s_t)

# %%
trainer.fit(lit_module, mesh_dm)

# %%
