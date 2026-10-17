import math
import numpy as np
import torch
from torch import nn
from torch.nn import ModuleList, ModuleDict
from .layers import ChebConv_Coma, Pool

from copy import copy
from typing import Sequence, Union, List

#TODO: Implement common parent class for encoder and decoder (GraphConvStack?), to capture common behaviour.

class PhaseBatchNorm1d(nn.Module):
    '''
    Batch norm for (batch, time, vertices, channels) inputs whose statistics and affine parameters
    are smooth functions of the cardiac phase theta (a Fourier series with n_harmonics harmonics):
    like the per-frame batch norm it centers and scales each phase separately (removing the
    population-average motion), but its (2 * n_harmonics + 1) * n_channels parameters don't depend
    on the number of frames, so the model works with any number of frames.

    Training: normalizes with each frame's batch statistics (over batch and vertices), as the
    per-frame batch norm does, and updates running Fourier fits of the mean and variance.
    Evaluation: uses the running fits evaluated at each frame's phase.
    '''

    def __init__(self, num_channels, n_harmonics=3, momentum=0.1, eps=1e-5):
        super(PhaseBatchNorm1d, self).__init__()
        self.n_harmonics, self.momentum, self.eps = n_harmonics, momentum, eps
        n_basis = 2 * n_harmonics + 1
        self.weight = nn.Parameter(torch.zeros(n_basis, num_channels))
        self.bias = nn.Parameter(torch.zeros(n_basis, num_channels))
        with torch.no_grad():
            self.weight[0] = 1.0  # constant gamma = 1 at init
        self.register_buffer("running_mean_coef", torch.zeros(n_basis, num_channels))
        self.register_buffer("running_var_coef", torch.zeros(n_basis, num_channels))
        self.running_var_coef[0] = 1.0
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def basis(self, phase):
        """(T,) phases in radians -> (T, 2K+1): [1, cos(theta), sin(theta), ..., cos(K theta), sin(K theta)]."""
        columns = [torch.ones_like(phase)]
        for k in range(1, self.n_harmonics + 1):
            columns += [torch.cos(k * phase), torch.sin(k * phase)]
        return torch.stack(columns, dim=1)

    def forward(self, x, phase=None):
        if x.dim() == 3:  # (batch, vertices, channels): a single frame at phase 0
            return self.forward(x.unsqueeze(1), phase=torch.zeros(1, device=x.device)).squeeze(1)
        n_timepoints = x.shape[1]
        if phase is None:  # equispaced frames over one cycle
            phase = 2 * math.pi * torch.arange(n_timepoints, device=x.device, dtype=x.dtype) / n_timepoints
        basis = self.basis(phase.to(device=x.device, dtype=x.dtype))  # (T, 2K+1)

        if self.training:
            mean = x.mean(dim=(0, 2))                  # (T, C)
            var = x.var(dim=(0, 2), unbiased=False)    # (T, C)
            with torch.no_grad():
                fit = torch.linalg.pinv(basis)         # least-squares Fourier fit over the frames
                n = x.shape[0] * x.shape[2]
                unbiased_var = var * n / max(n - 1, 1)
                self.running_mean_coef.lerp_(fit @ mean.detach(), self.momentum)
                self.running_var_coef.lerp_(fit @ unbiased_var.detach(), self.momentum)
                self.num_batches_tracked += 1
        else:
            mean = basis @ self.running_mean_coef
            var = (basis @ self.running_var_coef).clamp_min(self.eps)

        x_hat = (x - mean[None, :, None, :]) / torch.sqrt(var[None, :, None, :] + self.eps)
        gamma, beta = basis @ self.weight, basis @ self.bias  # (T, C)
        return x_hat * gamma[None, :, None, :] + beta[None, :, None, :]


class ParallelBatchNorm1d(nn.Module):
    '''
    Batch norm for (batch, [time,] vertices, channels) inputs. By default a 4D input is normalized
    per (frame, channel), so num_features = n_timeframes * n_channels and the module is tied to the
    number of frames. With across_time=True (num_features = n_channels) each channel is normalized
    with statistics pooled over batch, vertices and all frames, for any number of frames.
    '''

    def __init__(self, num_features, across_time=False):
        super(ParallelBatchNorm1d, self).__init__()
        self.batch_norm = nn.BatchNorm1d(num_features)
        self.across_time = across_time

    def forward(self, x):
        if len(x.size()) == 4 and self.across_time:
            batch_size, n_timepoints, n_vertices, n_channels = x.size()
            return self.batch_norm(x.reshape(-1, n_channels)).view(batch_size, n_timepoints, n_vertices, n_channels)

        # x shape: (batch_size, channels, sequence_length)
        if len(x.size()) == 4:
            batch_size, n_timepoints, n_vertices, n_channels = x.size()
            
            # Reshape to (batch_size * n_vertices, n_timepoints, n_channels)
            # Note: We need to reshape for batch_norm1d to work on the channel dimension
            x = x.permute(0, 2, 1, 3).contiguous().view(-1, n_timepoints * n_channels)
            
            # Apply batch normalization
            x = self.batch_norm(x)
            
            # Reshape back to (batch_size, n_timepoints, n_vertices, n_channels)
            x = x.view(batch_size, n_vertices, n_timepoints, n_channels).permute(0, 2, 1, 3).contiguous()

        else:
            batch_size, n_vertices, n_channels = x.size()
            
            # Reshape to (batch_size * n_vertices, n_timepoints, n_channels)
            # Note: We need to reshape for batch_norm1d to work on the channel dimension
            x = x.view(-1, n_channels)
            
            # Apply batch normalization
            x = self.batch_norm(x)
            
            # Reshape back to (batch_size, n_timepoints, n_vertices, n_channels)
            x = x.view(batch_size, n_vertices, n_channels).contiguous()

        return x


################# FULL AUTOENCODER #################

class Autoencoder3DMesh(nn.Module):

    def __init__(self, enc_config, dec_config):

        super(Autoencoder3DMesh, self).__init__()

        self.encoder = Encoder3DMesh(**enc_config)
        self.decoder = Decoder3DMesh(**dec_config)
        

    def forward(self, x):

        mu, logvar = self.encoder(x)
        # Add sampling if is_variational == True and it's in training mode
        # if self.is_variational and self.mode == "training":
        # z = mu + normal * exp(logvar)
        # else:
        z = mu
        x_hat = self.decoder(z)
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
    # "n_timeframes"
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
        template,
        adjacency_matrices: List[torch.Tensor],
        downsample_matrices: List[torch.Tensor],
        latent_dim: Union[None, int] = None,
        batch_normalization = True,
        n_timeframes = 1, 
        activation_layers="ReLU",
        batch_norm_across_time = False,
        batch_norm_phase_harmonics = 0):

        super(Encoder3DMesh, self).__init__()

        self.n_nodes = n_nodes
        self.phase_input = phase_input
        self.filters_enc = copy(num_conv_filters_enc)
        self.filters_enc.insert(0, num_features)
        self.K = cheb_polynomial_order
        self.batch_normalization = batch_normalization
        self.batch_norm_across_time = batch_norm_across_time
        self.batch_norm_phase_harmonics = batch_norm_phase_harmonics  # > 0: PhaseBatchNorm1d

        self.n_timeframes = n_timeframes
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
        

        if self.latent_dim is not None:
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
            if self.batch_normalization:
                if self.batch_norm_phase_harmonics > 0:
                    encoder[layer]["batch_normalization"] = PhaseBatchNorm1d(self.filters_enc[i+1], self.batch_norm_phase_harmonics)
                elif self.batch_norm_across_time:
                    encoder[layer]["batch_normalization"] = ParallelBatchNorm1d(self.filters_enc[i+1], across_time=True)
                else:
                    encoder[layer]["batch_normalization"] = ParallelBatchNorm1d(self.n_timeframes*self.filters_enc[i+1])
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
        x = x.reshape(x.shape[0], x.shape[1], -1)
        return x

    @staticmethod
    def build_from_dictionary(config_dict):
        enc_config = {k: v for k, v in config_dict.items() if k in ENCODER_ARGS}
        return Encoder3DMesh(**enc_config)


    # perform a forward pass only through the convolutional stack (not the FCN layer)
    def forward_conv_stack(self, x, preserve_graph_structure=True, phase=None):
        # phase: optional (T,) radians of x's frames, for PhaseBatchNorm1d (default: equispaced)
        
        # a "layer" here is: a graph convolution + pooling operation + activation function
        for i, layer in enumerate(self.layers): 
            
            if self.matrices["downsample"][i].device != x.device:
                self.matrices["downsample"][i] = self.matrices["downsample"][i].to(x.device)
            if self.matrices["A_edge_index"][i].device != x.device:
                self.matrices["A_edge_index"][i] = self.matrices["A_edge_index"][i].to(x.device)
            if self.matrices["A_norm"][i].device != x.device:
                self.matrices["A_norm"][i] = self.matrices["A_norm"][i].to(x.device)
  
            x = self.layers[layer]["graph_conv"](x, self.matrices["A_edge_index"][i], self.matrices["A_norm"][i])
            x = self.layers[layer]["pool"](x, self.matrices["downsample"][i])
            
            if 'batch_normalization' in self.layers[layer]:
                batch_norm = self.layers[layer]["batch_normalization"]
                x = batch_norm(x, phase=phase) if isinstance(batch_norm, PhaseBatchNorm1d) else batch_norm(x)
            
            x = self.layers[layer]["activation_function"](x)
        
        if not preserve_graph_structure:
            x = self.concatenate_graph_features(x)
            
        return x
    
    
    def forward(self, x):

        z = self.forward_conv_stack(x)
        
        if self.latent_dim is not None:
            mu = self.enc_lin_mu(x)
            log_var = self.enc_lin_var(x) if self._is_variational else None        
            z = {"mu": mu, "log_var": log_var}
        else:
            z = {"mu": z, "log_var": None}
        
        return z

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

class Decoder3DMesh(nn.Module):
    
    def __init__(self,
        num_features: int,
        n_layers: int,
        n_nodes: int,
        num_conv_filters_dec: Sequence[int],
        cheb_polynomial_order: int,
        latent_dim: int,
        is_variational: bool,
        template,
        upsample_matrices: List[torch.Tensor],
        adjacency_matrices: List[torch.Tensor],
        batch_normalization=True,
        n_timeframes=1,
        activation_layers="ReLU"):

        super(Decoder3DMesh, self).__init__()

        self.n_nodes = n_nodes
        self.filters_dec = copy(num_conv_filters_dec)
        self.filters_dec.insert(0, num_features)
        self.filters_dec = list(reversed(self.filters_dec))

        self.n_timeframes = n_timeframes
        self.K = cheb_polynomial_order
        self.batch_normalization = batch_normalization

        self.matrices = {}
        # Each decoder layer upsamples *then* graph-convolves, so layer i's
        # ChebConv needs the adjacency at the resolution produced by
        # upsample[i], not the pre-upsample one -- drop the coarsest (index
        # 0 pre-reversal / never reached post-upsample) entry accordingly.
        # Pre-existing off-by-one (silently tolerated by propagate(), which
        # infers node count from edge_index rather than requiring an exact
        # size match), not introduced by any later change to this class.
        A_edge_index, A_norm = self._build_adj_matrix(adjacency_matrices)
        self.matrices["A_edge_index"] = list(reversed(A_edge_index))[1:]
        self.matrices["A_norm"] = list(reversed(A_norm))[1:]
        self.matrices["upsample"] = list(reversed(upsample_matrices))

        self._n_features_before_z = self.matrices["upsample"][0].shape[1] * self.filters_dec[0]

        self._is_variational = is_variational
        self.latent_dim = latent_dim

        self.activation_layers = [activation_layers] * n_layers if isinstance(activation_layers, str) else activation_layers

        # Fully connected layer connecting the latent space layer with the first upsampling layer.
        self.dec_lin = torch.nn.Linear(self.latent_dim, self._n_features_before_z)

        self.layers = self._build_decoder()


    def _build_decoder(self):

        cheb_conv_layers = self._build_cheb_conv_layers(self.filters_dec, self.K)
        pool_layers = self._build_pool_layers(self.matrices["upsample"])
        activation_layers = self._build_activation_layers(self.activation_layers)

        decoder = ModuleDict()

        for i in range(len(cheb_conv_layers)):
            layer = f"layer_{i}"
            decoder[layer] = ModuleDict()
            decoder[layer]["activation_function"] = activation_layers[i]
            decoder[layer]["pool"] = pool_layers[i]
            if self.batch_normalization:
                decoder[layer]["batch_normalization"] = ParallelBatchNorm1d(self.n_timeframes*self.filters_dec[i])
            decoder[layer]["graph_conv"] = cheb_conv_layers[i]

        return decoder


    def _build_pool_layers(self, upsample_matrices:Sequence[np.array]):

        '''
        downsample_matrices: list of matrices binary matrices
        '''

        pool_layers = ModuleList()
        for i in range(len(upsample_matrices)):
            pool_layers.append(Pool())
        return pool_layers


    def _build_activation_layers(self, activation_type:Union[str, Sequence[str]]):

        '''
        activation_type: string or list of strings containing the name of a valid activation function from torch.functional
        '''

        activation_layers = ModuleList()

        for i in range(len(activation_type)):
            activ_fun = getattr(torch.nn.modules.activation, activation_type[i])()
            activation_layers.append(activ_fun)

        return activation_layers


    def _build_cheb_conv_layers(self, n_filters, K):
        # Chebyshev convolutions (decoder)
        cheb_dec = torch.nn.ModuleList([ChebConv_Coma(n_filters[0], n_filters[1], K[0])])
        for i in range(1, len(n_filters)-1):
            conv_layer = ChebConv_Coma(n_filters[i], n_filters[i+1], K[i])
            cheb_dec.extend([conv_layer])

        cheb_dec[-1].bias = None  # No bias for last convolution layer
        return cheb_dec


    def _build_adj_matrix(self, adjacency_matrices):
        adj_edge_index, adj_norm = zip(*[
            ChebConv_Coma.norm(adjacency_matrices[i]._indices(), self.n_nodes[i])
            for i in range(len(self.n_nodes))
        ])
        return list(adj_edge_index), list(adj_norm)


    @staticmethod
    def build_from_dictionary(config_dict):
        dec_config = {k: v for k, v in config_dict.items() if k in DECODER_ARGS}
        return Decoder3DMesh(**dec_config)


    def forward(self, x):

        x = self.dec_lin(x)
        batch_size = x.shape[0] if x.dim() == 2 else 1
        x = x.reshape(batch_size, -1, self.layers["layer_0"]["graph_conv"].in_channels)

        for i, layer in enumerate(self.layers):
            
            if self.matrices["upsample"][i].device != x.device:
                self.matrices["upsample"][i] = self.matrices["upsample"][i].to(x.device)
            if self.matrices["A_edge_index"][i].device != x.device:
                self.matrices["A_edge_index"][i] = self.matrices["A_edge_index"][i].to(x.device)
            if self.matrices["A_norm"][i].device != x.device:
                self.matrices["A_norm"][i] = self.matrices["A_norm"][i].to(x.device)

            x = self.layers[layer]["activation_function"](x)
            x = self.layers[layer]["pool"](x, self.matrices["upsample"][i])

            if 'batch_normalization' in self.layers[layer]:
                x = self.layers[layer]["batch_normalization"](x)

            x = self.layers[layer]["graph_conv"](x, self.matrices["A_edge_index"][i], self.matrices["A_norm"][i])

        return x
