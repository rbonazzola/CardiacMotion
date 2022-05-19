import numpy as np
import torch
from torch import nn
from torch.nn import ModuleList, ModuleDict
import torch.nn.functional as F
from .layers import ChebConv_Coma, Pool
from typing import Sequence, Union

from IPython import embed

class Autoencoder3DMesh(nn.Module):

    def __init__(self,
        phase_input: bool,
        num_conv_filters_enc: Sequence[int], num_features: int,
        cheb_polynomial_order: int,
        n_layers: int,
        n_nodes: int,
        is_variational: bool,
        latent_dim_content: int,
        adjacency_matrices,
        downsample_matrices,
        upsample_matrices,
        activation_layers="ReLU"):

        super(Autoencoder3DMesh, self).__init__()


        self.encoder = Encoder3DMesh(
            phase_input,
            num_conv_filters_enc,
            num_features,
            cheb_polynomial_order,
            n_layers,
            n_nodes,
            is_variational,
            latent_dim_content,
            adjacency_matrices,
            downsample_matrices,
            activation_layers
        )

        self.decoder = Decoder3DMesh(
            latent_dim_content,
            num_conv_filters_dec_c,
            num_conv_filters_dec_s,
            cheb_polynomial_order,
            upsample_matrices,
            adjacency_matrices
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat



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
        latent_dim_content: int, 
        adjacency_matrices,
        downsample_matrices,
        activation_layers="ReLU"):

        super(Encoder3DMesh, self).__init__()

        self.n_nodes = n_nodes
        self.phase_input = phase_input
        self.filters_enc = num_conv_filters_enc
        self.filters_enc.insert(0, num_features)
        self.K = cheb_polynomial_order
        self.adjacency_matrices = adjacency_matrices
        self.downsample_matrices = downsample_matrices

        self._n_features_before_z = self.downsample_matrices[-1].shape[0] * self.filters_enc[-1]

        self._is_variational = is_variational

        self.latent_dim = latent_dim_content

        self.A_edge_index, self.A_norm = self._build_adj_matrix()

        self.activation_layers = [activation_layers] * n_layers if isinstance(activation_layers, str) else activation_layers
        self.layers = self._build_encoder()

        # Fully connected layers connecting the last pooling layer and the latent space layer.
        self.enc_lin_mu = torch.nn.Linear(self._n_features_before_z, self.latent_dim)

        if self._is_variational:
            self.enc_lin_var = torch.nn.Linear(self._n_features_before_z, self.latent_dim)

    def _build_encoder(self):

        cheb_conv_layers = self._build_cheb_conv_layers(self.filters_enc, self.K)
        pool_layers = self._build_pool_layers(self.downsample_matrices)
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
            pool_layers.append(Pool(downsample_matrices[i]))
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
        # Chebyshev convolutions (encoder)

        #TOFIX: this should be specified in the docs.
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


    def _build_adj_matrix(self):
        adj_edge_index, adj_norm = zip(*[
            ChebConv_Coma.norm(self.adjacency_matrices[i]._indices(), self.n_nodes[i])
            for i in range(len(self.n_nodes))
        ])
        return list(adj_edge_index), list(adj_norm)

    
    def concatenate_graph_features(self, x):
        x = x.reshape(x.shape[0], self._n_features_before_z)
        return x


    def forward(self, x):

        # a "layer" here is: a graph convolution + pooling operation + activation function
        for i, layer in enumerate(self.layers): 
            x = self.layers[layer]["graph_conv"](x, self.A_edge_index[i], self.A_norm[i])
            x = self.layers[layer]["pool"](x)
            x = self.layers[layer]["activation_function"](x)
        
        x = self.concatenate_graph_features(x)
        
        mu = self.enc_lin_mu(x)
        log_var = self.enc_lin_var(x) if self._is_variational else None
        return mu, log_var


class Decoder3DMesh(nn.Module):
    
    def __init__(self,
        phase_input: bool,
        num_conv_filters_enc: Sequence[int], num_features: int,
        cheb_polynomial_order: int,
        n_layers: int,
        n_nodes: int,
        is_variational: bool,
        latent_dim_content: int,
        adjacency_matrices,
        upsample_matrices,
        activation_layers="ReLU"):

        super(Decoder3DMesh, self).__init__()

        self.n_nodes = n_nodes
        self.filters_dec = num_conv_filters_dec
        # self.filters_dec.insert(0, num_features)
        self.K = cheb_polynomial_order
        self.adjacency_matrices = adjacency_matrices
        self.A_edge_index, self.A_norm = self._build_adj_matrix()
        self.upsample_matrices = upsample_matrices

        self._n_features_before_z = self.upsample_matrices[-1].shape[0] * self.filters_dec[-1]

        self._is_variational = is_variational
        self.latent_dim = latent_dim_content

        self.activation_layers = [activation_layers] * n_layers if isinstance(activation_layers, str) else activation_layers

        self.layers = self._build_decoder()

        # Fully connected layers connecting the last pooling layer and the latent space layer.
        self.dec_lin = torch.nn.Linear(self.latent_dim, self._n_features_before_z)


    def _build_decoder(self):

        cheb_conv_layers = self._build_cheb_conv_layers(self.filters_dec, self.K)
        pool_layers = self._build_pool_layers(self.downsample_matrices)
        activation_layers = self._build_activation_layers(self.activation_layers)

        decoder = ModuleDict()

        for i in range(len(cheb_conv_layers)):
            layer = f"layer_{i}"
            decoder[layer] = ModuleDict()
            decoder[layer]["graph_conv"] = cheb_conv_layers[i]
            decoder[layer]["pool"] = pool_layers[i]
            decoder[layer]["activation_function"] = activation_layers[i]

        return decoder

    def _build_pool_layers(self, upsample_matrices:Sequence[np.array]):

        '''
        downsample_matrices: list of matrices binary matrices
        '''

        pool_layers = ModuleList()
        for i in range(len(upsample_matrices)):
            pool_layers.append(Pool(upsample_matrices[i]))
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
        cheb_dec.extend([
            ChebConv_Coma(
                n_filters[i],
                n_filters[i+1],
                K[i]
            ) for i in range(1, len(n_filters)-1)
        ])

        cheb_dec[-1].bias = None  # No bias for last convolution layer
        return cheb_dec


    def _build_adj_matrix(self):
        adj_edge_index, adj_norm = zip(*[
            ChebConv_Coma.norm(self.adjacency_matrices[i]._indices(), self.n_nodes[i])
            for i in range(len(self.n_nodes))
        ])
        return list(adj_edge_index), list(adj_norm)

    def forward(self, x):

        '''
        The decoder applies a phase embedding on the latent vector
        '''

        x = self.dec_lin()
        x = x.reshape(x.shape[0], cheb_dec[0].in_channels, -1)

        for i, layer in enumerate(self.layers):
            x = self.layers[layer]["activation_function"](x)
            x = self.layers[layer]["pool"](x)
            x = self.layers[layer]["graph_conv"](x, self.A_edge_index[i], self.A_norm[i])
        return x

