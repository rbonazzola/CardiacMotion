import numpy as np
import torch
from torch import nn

from typing import Sequence, Union, List, Literal
from copy import copy, deepcopy
import logging

from .Model3D import Encoder3DMesh, Decoder3DMesh
from .PhaseModule import PhaseTensor

from easydict import EasyDict

from .TemporalAggregators import (
  Mean_Aggregator, 
  DFT_Aggregator, 
  FCN_Aggregator
)

from utils.helpers import (
    get_coma_args,
    get_coma_matrices
)

logging.basicConfig(
  level=logging.INFO, 
  format='%(asctime)s - %(levelname)s - %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S'
)

BATCH_DIMENSION = 0
TIME_DIMENSION = 1
NODE_DIMENSION = 2
FEATURE_DIMENSION = 3

def _steal_attributes_from_child(self, child: str, attributes: Union[List[str], str]):

    '''
       Make attributes from an object's child visible from the (parent) object's namespace
       They can thus be accessed with a.c instead of a.b.c
    '''

    child = getattr(self, child)

    if isinstance(attributes, str):
        attributes = [attributes]

    for attribute in attributes:
        setattr(self, attribute, getattr(child, attribute))
    return self


def sampling(mu, log_var):
    std = torch.exp(0.5*log_var)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)


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


ENCODER_ARGS   = COMMON_ARGS + ["phase_input", "downsample_matrices", "num_conv_filters_enc", "latent_dim_c", "latent_dim_s"]
DECODER_C_ARGS = COMMON_ARGS + ["upsample_matrices", "num_conv_filters_dec_c", "latent_dim_content"]
DECODER_S_ARGS = COMMON_ARGS + ["upsample_matrices", "num_conv_filters_dec_s", "latent_dim_content", "latent_dim_style"]


class AutoencoderTemporalSequence(nn.Module):

    def __init__(self, 
                 encoder=None, decoder=None, 
                 enc_config=None, dec_c_config=None, dec_s_config=None, 
                 z_aggr_function="dft", n_timeframes=None, phase_embedding_method="exp", 
                 is_variational=False):

        super(AutoencoderTemporalSequence, self).__init__()
        
        self.n_timeframes = n_timeframes
        self.is_variational = is_variational

        if encoder is not None:
            self.encoder = encoder            
        else:
            self.encoder = EncoderTemporalSequence(
                enc_config, 
                z_aggr_function=z_aggr_function, 
                n_timeframes=n_timeframes,
                is_variational=is_variational
            )
            
        if decoder is not None:
            self.decoder = decoder
        else:            
            self.decoder = DecoderTemporalSequence(
                dec_c_config, 
                dec_s_config, 
                phase_embedding_method=phase_embedding_method,
                is_variational=is_variational
            )
                

    @classmethod
    def get_example_input_from_template(cls, mesh_template, n_timeframes):
        return torch.Tensor(mesh_template.v).unsqueeze(0).expand(n_timeframes, -1, -1).unsqueeze(0)


    @classmethod
    def build_from_config(cls, config, mesh_template, partition, n_timeframes, phase_embedding_method="exp_v1"):

        coma_matrices = get_coma_matrices(config, mesh_template, partition)
        (coma_args := get_coma_args(config)).update(coma_matrices)
      
        enc_config = EasyDict({k: v for k, v in coma_args.items() if k in ENCODER_ARGS})
        encoder = Encoder3DMesh(**enc_config, n_timeframes=n_timeframes)

        assert "latent_dim" not in enc_config or enc_config.latent_dim == config.network_architecture.latent_dim_c + config.network_architecture.latent_dim_s, f"{latent_dim=} but it should equal the sum of {config.network_architecture.latent_dim_c=} and {config.network_architecture.latent_dim_s=}"

        enc_config.latent_dim = config.network_architecture.latent_dim_c + config.network_architecture.latent_dim_s 
    
        x = cls.get_example_input_from_template(mesh_template, n_timeframes)
        h = encoder.forward_conv_stack(x, preserve_graph_structure=False)
        
        model = AutoencoderTemporalSequence(
            encoder = EncoderTemporalSequence(
                encoder3d = encoder, 
                z_aggr_function = FCN_Aggregator(features_in=n_timeframes * h.shape[-1], features_out=enc_config.latent_dim), 
                is_variational=coma_args.is_variational
            ), 
            decoder = DecoderTemporalSequence(
                decoder_content = DecoderContent.build_from_dictionary(coma_args),
                decoder_style   = DecoderStyle.build_from_dictionary(coma_args, phase_embedding_method=phase_embedding_method, n_timeframes=n_timeframes),
                is_variational=coma_args.is_variational),
            is_variational=coma_args.is_variational
        )

        return model

                    
    def forward(self, s_t):

        z = self.encoder(s_t)
        # z = self.sampling(mu, log_var)
        avg_s, shat_t = self.decoder(z)
                        
        return z, avg_s, shat_t

    
    def set_mode(self, mode: str):
        '''mode: "training" or "testing"'''
        self.mode = mode
        self.encoder.set_mode(mode)
        self.decoder.set_mode(mode)

    
##########################################################################################

class EncoderTemporalSequence(nn.Module):

    def __init__(self, encoder3d: Encoder3DMesh, z_aggr_function, phase_embedding=None, n_timeframes=None, is_variational=False):

        '''
        
        '''
        
        super(EncoderTemporalSequence, self).__init__()

        self.encoder_3d_mesh = encoder3d

        self = _steal_attributes_from_child(self, child="encoder_3d_mesh", attributes=["matrices"])

        self.z_aggr_function_mu = z_aggr_function        
        self.z_aggr_function_log_var = deepcopy(z_aggr_function)
        
        torch.nn.init.normal_(self.z_aggr_function_log_var.fcn.weight, 0, 1e-5)
        torch.nn.init.normal_(self.z_aggr_function_mu.fcn.weight, 0, 1e-5)
        
        self.phase_embedding = phase_embedding
        self.is_variational = is_variational


    def set_mode(self, mode: str):
        '''
        params:
          mode: "training" or "testing"
        '''
        self.mode = mode


    def encoder(self, x):
                
        self.n_timeframes = x.shape[TIME_DIMENSION]

        # Iterate through time points
        # bottleneck_t = [ self.encoder_3d_mesh(x[:, i, :]) for i in range(self.n_timeframes) ]
        # mu = [ bottleneck["mu"] for bottleneck in bottleneck_t ]

        # If one element (and therefore all elements) are None, replace the whole thing with None
        # log_var = [ bottleneck["log_var"] for bottleneck in bottleneck_t ] if bottleneck_t[0]["log_var"] is not None else None

        # mu = torch.cat(mu).reshape(-1, self.n_timeframes, self.latent_dim)
        
        h = self.encoder_3d_mesh.forward_conv_stack(x, preserve_graph_structure=False)
        mu = self.z_aggr_function_mu(h)
        
        if self.is_variational:
            logging.debug("Generating mu and log_var.")
            log_var = self.z_aggr_function_log_var(h)
        else:
            log_var = None
            
        bottleneck = {"mu": mu, "log_var": log_var}
        return bottleneck


    def forward(self, x):
        return self.encoder(x)

  
##########################################################################################

PHASE_EMBEDDINGS = Literal[
    "inverse_dft", "dft", 
    "concatenation", "concat", 
    "exponential_v1", "exp_v1", "exp", 
    "exponential_v2", "exp_v2"
] 

import inspect

class DecoderContent(Decoder3DMesh):
    
    def __init__(self, decoder_c_config):
        
        decoder_c_config = copy(decoder_c_config)
        decoder_c_config["num_conv_filters_dec"] = decoder_c_config.pop("num_conv_filters_dec_c")
        decoder_c_config["latent_dim"] = decoder_c_config.pop("latent_dim_content")
        decoder_c_config["n_timeframes"] = 1
        
        super(DecoderContent, self).__init__(**decoder_c_config)

    @classmethod
    def build_from_dictionary(cls, config_dict):
        dec_config = { k: v for k, v in config_dict.items() if k in DECODER_C_ARGS }
        return cls(dec_config)
                 
            
class DecoderStyle(nn.Module):

    def __init__(self, decoder_config: dict, 
                 phase_embedding_method: PHASE_EMBEDDINGS = "exp", 
                 n_timeframes: Union[int, None]=None):

        super(DecoderStyle, self).__init__()

        decoder_config = copy(decoder_config)
        # self.n_timeframes = decoder_config.pop("n_timeframes")
        self.n_timeframes = n_timeframes
        self.phase_embedding = self._get_phase_embedding(phase_embedding_method, self.n_timeframes)

        decoder_config = copy(decoder_config)
        decoder_config["latent_dim"] = decoder_config.pop("latent_dim_content") + 2 * decoder_config.pop("latent_dim_style")
        decoder_config["num_conv_filters_dec"] = decoder_config.pop("num_conv_filters_dec_s")

        self.decoder_3d = Decoder3DMesh(**decoder_config)


    def  _get_phase_embedding(self, phase_embedding_method, n_timeframes):

        if phase_embedding_method.lower() in ["inverse_dft", "dft"]:
            raise NotImplementedError

        elif phase_embedding_method.lower() in ["concatenation", "concat"]:
            raise NotImplementedError

        elif phase_embedding_method.lower() in ["exponential_v1", "exp_v1", "exp"]:
            return PhaseTensor(version="version_1")

        elif phase_embedding_method.lower() in ["exponential_v2", "exp_v2"]:
            return PhaseTensor(version="version_2")

        else:
            raise ValueError(f"Method of phase embedding {phase_embedding_method} has not been recognised.")


    def _process_one_timeframe(self, z_c, phased_z_s, t):

        z_s_t = phased_z_s[:, t, ...]
        z = torch.cat([z_c, z_s_t], axis=-1)
        s_t = self.decoder_3d(z)
        s_t = s_t.unsqueeze(1)
        return s_t


    def forward(self, z_c, z_s, n_timeframes):
        
        phased_z_s = z_s.unsqueeze(TIME_DIMENSION).repeat(1, self.n_timeframes, *[1 for x in z_s.shape[1:]])
        phased_z_s = self.phase_embedding(phased_z_s)
        s_out = [ self._process_one_timeframe(z_c, phased_z_s, t) for t in range(n_timeframes) ]
        s_out = torch.cat(s_out, dim=1)
                
        return s_out
    

    @classmethod
    def build_from_dictionary(cls, config_dict, phase_embedding_method, n_timeframes):
        dec_config = {k: v for k, v in config_dict.items() if k in DECODER_S_ARGS}
        return cls(dec_config, phase_embedding_method, n_timeframes)
      
            
class DecoderTemporalSequence(nn.Module):

    # def __init__(self, decoder_c_config, decoder_s_config, phase_embedding_method: PHASE_EMBEDDINGS = "exp", n_timeframes=None):
    def __init__(self, decoder_content, decoder_style, is_variational):

        super(DecoderTemporalSequence, self).__init__()

        # self.template_mesh = decoder_c_config["template"]
        
        self.latent_dim_content = decoder_content.latent_dim
        
        #TOFIX: 
        self.latent_dim_style = (decoder_style.decoder_3d.latent_dim - self.latent_dim_content) / 2
        
        self.decoder_content = decoder_content
        self.decoder_style = decoder_style
        self.is_variational = is_variational
        
        self = _steal_attributes_from_child(self, child="decoder_content", attributes=["matrices"])
    

    def set_mode(self, mode: str):
        '''
        params:
          mode: "training" or "testing"
        '''
        self._mode = mode


    def forward(self, z):

        bottleneck = self._partition_z(z["mu"], z["log_var"])
        
        mu_c, mu_s = bottleneck["mu_c"], bottleneck["mu_s"]
        
        if self.is_variational and self._mode == "training":
            logging.debug("Sampling z.")
            log_var_c, log_var_s = bottleneck["log_var_c"], bottleneck["log_var_s"]
            z_c = sampling(mu_c, log_var_c)
            z_s = sampling(mu_s, log_var_s)
        else:
            z_c, z_s = mu_c, mu_s
        
        # Compute time-averaged shape
        avg_shape = self.decoder_content(z_c)
        
        # Compute deformation with respect to time-averaged shape
        def_field_t = self.decoder_style(z_c, z_s, self.decoder_style.n_timeframes)        
        
        # Compute shape at time t
        shape_t = avg_shape.unsqueeze(TIME_DIMENSION) + def_field_t
        
        return avg_shape, shape_t


    def _partition_z(self, mu, log_var=None):

        bottleneck = {
            "mu_c": mu[:, :self.latent_dim_content],
            "mu_s": mu[:, self.latent_dim_content:]
        }

        if log_var is not None:
            bottleneck.update({
                "log_var_c": log_var[:, :self.latent_dim_content],
                "log_var_s": log_var[:, self.latent_dim_content:]
            })

        return bottleneck