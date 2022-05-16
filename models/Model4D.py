import torch
from torch import nn
from torch.nn import ModuleList, ModuleDict
import torch.nn.functional as F

from IPython import embed

import numpy as np

from .layers import ChebConv_Coma, Pool
from .PhaseModule import PhaseTensor
from .TemporalAggregators import Mean_Aggregator, DFT_Aggregator, FCN_Aggregator

from typing import Sequence, Union

class Coma4D_C_and_S(torch.nn.Module):

    def __init__(self,
          num_features,
          num_conv_filters_enc,
          polygon_order,
          latent_dim_content,
          latent_dim_style,
          is_variational,
          downsample_matrices,
          upsample_matrices,
          adjacency_matrices,
          template,
          n_nodes,
          z_aggr_function,
          phase_input,
          n_timeframes=None,
          mode="testing"
    	):

        self.encoder = Encoder(
            n_layers,
            phase_input,
            num_conv_filters_enc,
            num_features,
            adjacency_matrices,
            polygon_order,
            is_variational,
            latent_dim_content,
            latent_dim_style,
            downsample_matrices,
            z_aggr_function
        )

        self.decoder = Decoder(
            latent_dim_content,
            latent_dim_style,
            n_layers,
            num_conv_filters_dec_c,
            num_conv_filters_dec_s,
            upsample_matrices,
            adjacency_matrices
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


class EncoderTemporalSequence(nn.Module):


    def __init__(self, encoder_config, z_aggr_function):

        self.encoder_3d_mesh = Encoder3DMesh(**encoder_config)
        self.z_aggr_function = self._get_z_aggr_function(z_aggr_function)


    def _get_z_aggr_function(self, z_aggr_function):

        if z_aggr_function == "mean":
            z_aggr_function = Mean_Aggregator()

        elif z_aggr_function.lower() == "fcn" or z_aggr_function.lower() == "fully_connected":
            self.n_timeframes = n_timeframes
            z_aggr_function = FCN_Aggregator(
                features_in=n_timeframes * (self.z_c + self.z_s),
                features_out=(self.z_c + self.z_s)
            )

        elif z_aggr_function.lower() == "dft" or z_aggr_function.lower() == "discrete_fourier_transform":
            self.n_timeframes = n_timeframes
            z_aggr_function = DFT_Aggregator(
                features_in=(n_timeframes // 2 + 1) * 2 * (self.z_c + self.z_s),
                features_out=(self.z_c + self.z_s)
            )

        return z_aggr_function


    def encoder(self, x):

        self.n_timeframes = x.shape[1]

        x = self.encoder_3d_mesh(x)
        x = self.concatenate_graph_features(x)

        mu, log_var = [], []

        # Iterate through time points
        for i in range(self.n_timeframes):
            _mu = self.enc_lin_mu(x[:, i, :])
            mu.append(_mu)

            if self._is_variational and self._mode == "training":
                log_var = self.enc_lin_var(x[:, i, :])
                log_var.append(_log_var)
            else:
                log_var = None

        mu = torch.cat(mu).reshape(-1, self.n_timeframes, self.z_c + self.z_s)
        mu = self.z_aggr_function(mu)

        if log_var is not None:
            log_var_t = torch.cat(log_var).reshape(-1, self.n_timeframes, self.z_c + self.z_s)
            log_var = self.z_aggr_function(log_var_t)

        bottleneck =  self._partition_z(mu, log_var)
        return bottleneck


    def _partition_z(self, mu, log_var=None):

        mu_c = mu[:, :self.z_c]
        mu_s = mu[:, self.z_c:]
        bottleneck = {"mu_s": mu_s, "mu_c": mu_c}

        if log_var is not None:
            log_var_c = log_var[:, :self.z_c]
            log_var_s = log_var[:, self.z_c:]
            bottleneck["log_var_c"] = log_var_c
            bottleneck["log_var_s"] = log_var_s

        return bottleneck


