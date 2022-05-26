import numpy as np
import torch
from torch import nn

from models.Model3D import Encoder3DMesh, Decoder3DMesh
from .PhaseModule import PhaseTensor
from .TemporalAggregators import Mean_Aggregator, DFT_Aggregator, FCN_Aggregator

from typing import Sequence, Union, List
from copy import copy
from IPython import embed

BATCH_DIMENSION = 0
TIME_DIMENSION = 1
NODE_DIMENSION = 2
FEATURE_DIMENSION = 3

def _steal_attributes_from_child(self, child: str, attributes: Union[List[str], str]):

    '''
       Make attributes from an object's child visible from the object's namespace
    '''

    child = getattr(self, child)
    for attribute in attributes:
        setattr(self, attribute, getattr(child, attribute))
    return self

COMMON_ARGS = [
    "num_features",
    "n_layers",
    "n_nodes",
    "cheb_polynomial_order",
    "is_variational",
    "adjacency_matrices",
    "activation_layers",
]

##########################################################################################

ENCODER_ARGS = copy(COMMON_ARGS)
ENCODER_ARGS.extend([
  "phase_input",
  "downsample_matrices",
  "num_conv_filters_enc",
  "latent_dim_content",
  "latent_dim_style"
])


class EncoderTemporalSequence(nn.Module):

    def __init__(self, encoder_config, z_aggr_function, phase_embedding=None, n_timeframes=None):

        super(EncoderTemporalSequence, self).__init__()
        encoder_config = copy(encoder_config)
        encoder_config["latent_dim"] = encoder_config.pop("latent_dim_content") + encoder_config.pop("latent_dim_style")
        self.encoder_3d_mesh = Encoder3DMesh(**encoder_config)

        self = _steal_attributes_from_child(self, child="encoder_3d_mesh", attributes=["downsample_matrices", "adjacency_matrices", "A_edge_index", "A_norm", "latent_dim"])

        self.z_aggr_function = self._get_z_aggr_function(z_aggr_function, n_timeframes)
        self.phase_embedding = phase_embedding


    def _get_z_aggr_function(self, z_aggr_function, n_timeframes=None):

        if z_aggr_function == "mean":
            if phase_embedding is None:
                exit("The temporal aggregation cannot be the mean if phase information is not embedded into the input meshes.")
            z_aggr_function = Mean_Aggregator()

        elif z_aggr_function.lower() in {"fcn", "fully_connected"}:
            self.n_timeframes = n_timeframes
            z_aggr_function = FCN_Aggregator(
                features_in=n_timeframes * self.latent_dim,
                features_out=(self.latent_dim)
            )

        elif z_aggr_function.lower() in {"dft", "discrete_fourier_transform"}:
            self.n_timeframes = n_timeframes
            z_aggr_function = DFT_Aggregator(
                features_in=(n_timeframes // 2 + 1) * 2 * (self.latent_dim),
                features_out=(self.latent_dim)
            )

        return z_aggr_function


    def set_mode(self, mode: str):
        '''
        params:
          mode: "training" or "testing"
        '''
        self._mode = mode


    def encoder(self, x):

        self.n_timeframes = x.shape[1]

        # Iterate through time points
        bottleneck_t = [ self.encoder_3d_mesh(x[:, i, :]) for i in range(self.n_timeframes) ]
        mu = [ bottleneck["mu"] for bottleneck in bottleneck_t ]

        # If one element (and therefore all elements) are None, replace the whole thing with None
        log_var = [ bottleneck["log_var"] for bottleneck in bottleneck_t ] if bottleneck_t[0]["log_var"] is not None else None

        mu = torch.cat(mu).reshape(-1, self.n_timeframes, self.latent_dim)
        mu = self.z_aggr_function(mu)

        if log_var is not None:
            log_var_t = torch.cat(log_var).reshape(-1, self.n_timeframes, self.latent_dim)
            log_var = self.z_aggr_function(log_var_t)

        bottleneck = {"mu": mu, "log_var": log_var}
        return bottleneck


    def forward(self, x):
        return self.encoder(x)


##########################################################################################

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

class DecoderStyle(nn.Module):

    def __init__(self, decoder_config: dict, phase_embedding_method: str, n_timeframes: Union[int, None]=None):

        super(DecoderStyle, self).__init__()

        decoder_config = copy(decoder_config)
        self.n_timeframes = decoder_config.pop("n_timeframes")
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


class DecoderTemporalSequence(nn.Module):

    def __init__(self, decoder_c_config, decoder_s_config, phase_embedding_method, n_timeframes=None):

        super(DecoderTemporalSequence, self).__init__()

        decoder_c_config = copy(decoder_c_config)
        decoder_c_config["num_conv_filters_dec"] = decoder_c_config.pop("num_conv_filters_dec_c")
        decoder_c_config["latent_dim"] = decoder_c_config.pop("latent_dim_content")

        self.latent_dim_content = decoder_c_config["latent_dim"]
        self.latent_dim_style = decoder_s_config["latent_dim_style"]

        self.decoder_content = Decoder3DMesh(**decoder_c_config)
        self.decoder_style = DecoderStyle(decoder_s_config, phase_embedding_method, n_timeframes)

        self = _steal_attributes_from_child(self, child="decoder_content", attributes=["upsample_matrices", "adjacency_matrices", "A_edge_index", "A_norm"])


    def forward(self, z):

        bottleneck = self._partition_z(z["mu"], z["log_var"])
        z_c, z_s = bottleneck["mu_c"], bottleneck["mu_s"]
        avg_shape = self.decoder_content(z_c)
        def_field_t = self.decoder_style(z_c, z_s, self.decoder_style.n_timeframes)
        shape_t = avg_shape.unsqueeze(TIME_DIMENSION) + def_field_t
        return avg_shape, shape_t


    def _partition_z(self, mu, log_var=None):

        bottleneck = {"mu_c": mu[:, :self.latent_dim_content], "mu_s": mu[:, self.latent_dim_content:]}

        if log_var is not None:
            bottleneck.update({
                "log_var_c": log_var[:, :self.latent_dim_content],
                "log_var_s": log_var[:, self.latent_dim_content:]
            })

        return bottleneck


class AutoencoderTemporalSequence(nn.Module):

    def __init__(self, enc_config, dec_c_config, dec_s_config, z_aggr_function="dft", n_timeframes=None, phase_embedding_method="exp"):

        super(AutoencoderTemporalSequence, self).__init__()
        self.encoder = EncoderTemporalSequence(enc_config, z_aggr_function=z_aggr_function, n_timeframes=n_timeframes)
        self.decoder = DecoderTemporalSequence(dec_c_config, dec_s_config, phase_embedding_method=phase_embedding_method)


    def forward(self, s_t):

        z = self.encoder(s_t)
        avg_s, shat_t = self.decoder(z)
        return avg_s, shat_t
