import numpy as np
import torch
from torch import nn
from models.Model3D import Encoder3DMesh

from .PhaseModule import PhaseTensor
from .TemporalAggregators import Mean_Aggregator, DFT_Aggregator, FCN_Aggregator
from typing import Sequence, Union
from IPython import embed

class EncoderTemporalSequence(nn.Module):

    def __init__(self, encoder_config, z_aggr_function, n_timeframes=None):

        super(EncoderTemporalSequence, self).__init__()
        self.latent_dim = encoder_config["latent_dim_content"] # + encoder_config["latent_dim_style"]
        self.encoder_3d_mesh = Encoder3DMesh(**encoder_config)

        self.downsample_matrices = self.encoder_3d_mesh.downsample_matrices
        self.adjacency_matrices = self.encoder_3d_mesh.adjacency_matrices
        self.A_edge_index, self.A_norm = self.encoder_3d_mesh.A_edge_index, self.encoder_3d_mesh.A_norm

        self.z_aggr_function = self._get_z_aggr_function(z_aggr_function, n_timeframes)


    def _get_z_aggr_function(self, z_aggr_function, n_timeframes=None):

        if z_aggr_function == "mean":
            z_aggr_function = Mean_Aggregator()

        elif z_aggr_function.lower() == "fcn" or z_aggr_function.lower() == "fully_connected":
            self.n_timeframes = n_timeframes
            z_aggr_function = FCN_Aggregator(
                features_in=n_timeframes * self.latent_dim,
                features_out=(self.latent_dim)
            )

        elif z_aggr_function.lower() == "dft" or z_aggr_function.lower() == "discrete_fourier_transform":
            self.n_timeframes = n_timeframes
            z_aggr_function = DFT_Aggregator(
                features_in=(n_timeframes // 2 + 1) * 2 * (self.latent_dim),
                features_out=(self.latent_dim)
            )

        return z_aggr_function


    def set_mode(self, mode):
        '''
        params:
          mode: "training" or "testing"
        '''
        self._mode = mode


    def encoder(self, x):

        self.n_timeframes = x.shape[1]

        mu, log_var = [], []

        # Iterate through time points
        for i in range(self.n_timeframes):
            _mu, _log_var = self.encoder_3d_mesh(x[:, i, :])
            mu.append(_mu)

            if _log_var is not None:
                log_var.append(_log_var)

        mu = torch.cat(mu).reshape(-1, self.n_timeframes, self.latent_dim)
        mu = self.z_aggr_function(mu)

        if _log_var is not None:
            log_var_t = torch.cat(log_var).reshape(-1, self.n_timeframes, self.latent_dim)
            log_var = self.z_aggr_function(log_var_t)

        # bottleneck =  self._partition_z(mu, log_var)
        bottleneck = {"mu": mu, "log_var": log_var}
        return bottleneck


    def forward(self, x):

        return self.encoder(x)


class DecoderStyle(nn.Module):

    def __init__(self, decoder_config, phase_embedding_method, n_timeframes=None):

        self.n_nodes = n_nodes
        self.n_layers = n_layers

        self.filters_dec_s = num_conv_filters_dec_s
        self.filters_dec_s.insert(0, num_features)

        self.K = polygon_order

        self.cheb_dec_s = self._build_decoder(self.filters_dec_s, self.K)

        self.dec_lin_s = torch.nn.Linear(
            self.latent_dim,
            self.filters_dec_s[-1] * self.upsample_matrices[-1].shape[1]
        )


    def _build_decoder(self, n_filters, K):

        # Chebyshev deconvolutions (decoder)
        cheb_dec = torch.nn.ModuleList([
            ChebConv_Coma(
                n_filters[-i - 1],
                n_filters[-i - 2],
                K[i]
            ) for i in range(len(n_filters) - 1)
        ])
        cheb_dec[-1].bias = None  # No bias for last convolution layer
        return cheb_dec


class DecoderTemporalSequence(nn.Module):


    def __init__(self, decoder_config, phase_embedding_method, n_timeframes=None):

        super(DecoderTemporalSequence, self).__init__()
        self.latent_dim = encoder_config["latent_dim_content"] # + encoder_config["latent_dim_style"]
        self.decoder_3d_mesh = Decoder3DMesh(**decoder_config)

        self.downsample_matrices = self.decoder_3d_mesh.downsample_matrices
        self.adjacency_matrices = self.decoder_3d_mesh.adjacency_matrices
        self.A_edge_index, self.A_norm = self.encoder_3d_mesh.A_edge_index, self.encoder_3d_mesh.A_norm

        self.phase_embedding = self._get_phase_embedding(phase_embedding_method, n_timeframes)

        self.decoder_content = Decoder3DMesh()
        self.decoder_style = DecoderStyle()



   def forward(self, z):

       z_c, z_s = self.partition_z(z)

       phased_z_s = self.phase_embedding(z)

       avg_shape = self.decoder_content(z_c)
       def_field_t = self.decoder_style(phased_z_s)

       shape_t = avg_shape + def_field_t

       return avg_shape, shape_t


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

   def _get_phase_embedding(self):
       pass




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
