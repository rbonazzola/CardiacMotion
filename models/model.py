import torch
import torch.nn.functional as F
from .layers import ChebConv_Coma, Pool
from IPython import embed
import numpy as np


__author__ = ['Priyanka Patel', 'Rodrigo Bonazzola']

class Coma(torch.nn.Module):

    def __init__(self, 
          n_layers,
          num_features, 
          num_conv_filters,
          polygon_order,
          latent_dim,
          is_variational,
          downsample_matrices, 
          upsample_matrices, 
          adjacency_matrices, 
          n_nodes,
          mode="testing",
          z_aggr_function=lambda x: torch.Tensor.mean(x, axis=1)
        ):
        
        super(Coma, self).__init__()

        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.filters = num_conv_filters
        self.filters.insert(0, num_features)

        self.K = polygon_order
        self.z = latent_dim
        self.downsample_matrices = downsample_matrices
        self.upsample_matrices = upsample_matrices
        self.adjacency_matrices = adjacency_matrices
        self._n_features_before_z = self.downsample_matrices[-1].shape[0] * self.filters[-1]

        self._mode = mode
        self._is_variational = is_variational
        
        self.A_edge_index, self.A_norm = self._build_adj_matrix()
        self.cheb_enc, self.cheb_dec = self._build_encoder(), self._build_decoder()                
        self.pool = Pool()
        
        # Fully connected layers connecting the last pooling layer and the latent space layer.
        self.enc_lin_mu = torch.nn.Linear(self._n_features_before_z, self.z)        
        if self._is_variational:            
            self.enc_lin_var = torch.nn.Linear(self._n_features_before_z, self.z)

        self.dec_lin = torch.nn.Linear(self.z, self.filters[-1]*self.upsample_matrices[-1].shape[1])

        self.z_aggr_function = z_aggr_function

        self.reset_parameters()

    def _build_encoder(self):
        # Chebyshev convolutions (encoder)
        cheb_enc = torch.nn.ModuleList([
            ChebConv_Coma(
                self.filters[i],
                self.filters[i+1],
                self.K[i]
            ) for i in range(len(self.filters)-1)
        ])
        return cheb_enc

    def _build_decoder(self):
        # Chebyshev deconvolutions (decoder)
        cheb_dec = torch.nn.ModuleList([
            ChebConv_Coma(
                self.filters[-i-1],
                self.filters[-i-2],
                self.K[i]
            ) for i in range(len(self.filters)-1)
        ])
        cheb_dec[-1].bias = None  # No bias for last convolution layer
        return cheb_dec

    def _build_adj_matrix(self):
        adj_edge_index, adj_norm = zip(*[
            ChebConv_Coma.norm(self.adjacency_matrices[i]._indices(), self.n_nodes[i])
            for i in range(len(self.n_nodes))
        ])
        return adj_edge_index, adj_norm


    def set_mode(self, mode):
        '''
        params:
          mode: "training" or "testing"
        '''
        self._mode = mode

    def reset_parameters(self):
        if self._is_variational:
            torch.nn.init.normal_(self.enc_lin_mu.weight, 0, 0.1)
            torch.nn.init.normal_(self.enc_lin_var.weight, 0, 0.1)
        else:
            torch.nn.init.normal_(self.enc_lin.weight, 0, 0.1)
        torch.nn.init.normal_(self.dec_lin.weight, 0, 0.1)


    def _sample(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) 


    def forward(self, x):

        batch_size = x.shape[0]
        time_frames = x.shape[1]
        
        x = x.reshape(batch_size, time_frames, -1, self.filters[0])
        
        if self._is_variational and self._mode == "training":            
            self.mu, self.log_var = self.encoder(x)
            z = self._sample(self.mu, self.log_var)
        else:
            z = self.encoder(x)

        x = self.decoder(z)

        x = x.reshape(-1, self.filters[0])
        return z, x


    def encoder(self, x):

        '''

        '''

        for i in range(self.n_layers):
            x = self.cheb_enc[i](x, self.A_edge_index[i], self.A_norm[i])
            x = F.relu(x)
            x = self.pool(x, self.downsample_matrices[i])

        timeframes = x.shape[1]
        x = x.reshape(x.shape[0], timeframes, self._n_features_before_z)
        
        mu, log_var = [], []
        for i in range(x.shape[1]):
            mu.append(self.enc_lin_mu(x[:,i,:]))
            if self._is_variational and self._mode == "training":
                log_var.append(self.enc_lin_var(x[:,i,:]))
        
        mu = torch.cat(mu)
        mu = mu.reshape(-1, time_frames, self.z)
        log_var
        mu = self.z_aggr_function(mu)

        if self._is_variational and self._mode == "training":
            log_var = torch.cat(log_var)
            log_var = log_var.reshape(-1, time_frames, self.z)
            log_var = self.z_aggr_function(log_var)
            return mu, log_var
        else:
            return mu

    def decoder(self, x):
        
        '''
        The decoder applies a phase embedding on the latent vector
        '''

        x = F.relu(self.dec_lin(x))        
        x = x.reshape(x.shape[0], cheb_dec[0].in_channels, -1)
        
        for i in range(self.n_layers-1):
            x = self.pool(x, self.upsample_matrices[-i-1])
            x = self.cheb_dec[i](x, self.A_edge_index[self.n_layers-i-1], self.A_norm[self.n_layers-i-1])
            x = F.relu(x)
        x = self.cheb_dec[-1](x, self.A_edge_index[-1], self.A_norm[-1])
        return x


class Coma4D(Coma):


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dec_lin = torch.nn.Linear(2*self.z, self.filters[-1]*self.upsample_matrices[-1].shape[1])


    def encoder(self, x):

        for i in range(self.n_layers):
            x = self.cheb_enc[i](x, self.A_edge_index[i], self.A_norm[i])
            x = F.relu(x)
            x = self.pool(x, self.downsample_matrices[i])

        self.n_timeframes = x.shape[1]
        x = x.reshape(x.shape[0], self.n_timeframes, self._n_features_before_z)
        
        mu, log_var = [], []
        for i in range(x.shape[1]):
            mu.append(self.enc_lin_mu(x[:,i,:]))
            if self._is_variational and self._mode == "training":
                log_var.append(self.enc_lin_var(x[:,i,:]))
        
        mu = torch.cat(mu)
        mu = mu.reshape(-1, self.n_timeframes, self.z)        
        mu = self.z_aggr_function(mu)

        if self._is_variational and self._mode == "training":
            log_var = torch.cat(log_var)
            log_var = log_var.reshape(-1, self.n_timeframes, self.z)
            log_var = self.z_aggr_function(log_var)
            return mu, log_var
        else:
            return mu


    def decoder(self, z):
    
        z_t = self.phase_embedding(z)
        # z_t = self.phase_embedding(z)
        s_out = []

        for t in range(self.n_timeframes):

            x = self.dec_lin(z_t[:,t,:])
            x = F.relu(x)        

            # x = x.reshape(-1, self.cheb_dec[0].in_channels)
            x = x.reshape(x.shape[0], -1, self.cheb_dec[0].in_channels)
            for i in range(self.n_layers-1):
                x = self.pool(x, self.upsample_matrices[-i-1])
                x = self.cheb_dec[i](x, self.A_edge_index[self.n_layers-i-1], self.A_norm[self.n_layers-i-1])
                x = F.relu(x)
            x = self.pool(x, self.upsample_matrices[0])
            x = self.cheb_dec[-1](x, self.A_edge_index[-1], self.A_norm[-1])
            
            #TODO: comment this
            x = x.unsqueeze(1)
            s_out.append(x)

        s_out = torch.cat(s_out, dim=1)
        return s_out


    def forward(self, x):

        batch_size = x.shape[0]
        time_frames = x.shape[1]
                
        x = x.reshape(batch_size, time_frames, -1, self.filters[0])
        
        if self._is_variational and self._mode == "training":            
            self.mu, self.log_var = self.encoder(x)            
            z = self._sample(self.mu, self.log_var)
            if torch.isinf(z).any():
                self.mu, self.log_var = self.mu/1000, self.log_var/1000
                z = self._sample(self.mu, self.log_var)
        else:
            z = self.encoder(x)

        x = self.decoder(z)

        return (self.mu, self.log_var), x


    def phase_embedding(self, z):

        '''
        params:
          z: a batched vector (N x M)

        returns:
          a phase-aware vector (N x T x 2M)
        '''

        exp_it = []

        for i in range(self.n_timeframes):
            phase = 2*np.pi*i/self.n_timeframes
            exp_it.append([np.sin(phase), np.cos(phase)])

        exp_it = np.array(exp_it)
        exp_it = np.expand_dims(exp_it, axis=(0,2))
        exp_it = torch.Tensor(exp_it)
        exp_it = exp_it.reshape(self.n_timeframes, 1, 2)

        # trick to achieve the desired dimensions
        z = z.reshape(-1, 1, self.z, 1)
        z_t = torch.matmul(z,exp_it)

        z_t = z_t.reshape(*z_t.shape[:2], -1)

        return z_t