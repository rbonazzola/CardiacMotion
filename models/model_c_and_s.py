import torch
from torch import nn
import torch.nn.functional as F
from .layers import ChebConv_Coma, Pool
from .PhaseModule import PhaseTensor
from IPython import embed

class Coma4D_C_and_S(torch.nn.Module):

    def __init__(self,
          n_layers,
          num_features, 
          num_conv_filters_enc,
          num_conv_filters_dec_c,
          num_conv_filters_dec_s,
          polygon_order,
          latent_dim_content,
          latent_dim_style,
          is_variational,
          downsample_matrices, 
          upsample_matrices, 
          adjacency_matrices,
          template,
          n_nodes,
          mode="testing",
          phase_input=True,
          z_aggr_function=lambda x: torch.Tensor.mean(x, axis=1)
    	):
        
        super(Coma4D_C_and_S, self).__init__()

        self.n_nodes = n_nodes
        self.n_layers = n_layers

        self.phase_input = phase_input        

        self.filters_enc = num_conv_filters_enc
        self.filters_dec_c = num_conv_filters_dec_c
        self.filters_dec_s = num_conv_filters_dec_s

        self.filters_enc.insert(0, num_features)
        self.filters_dec_c.insert(0, num_features)
        self.filters_dec_s.insert(0, num_features)

        self.K = polygon_order
        self.z_c = latent_dim_content
        self.z_s = latent_dim_style

        self.downsample_matrices = downsample_matrices
        self.upsample_matrices = upsample_matrices
        self.adjacency_matrices = adjacency_matrices
        self.template_mesh = template

        self._n_features_before_z = self.downsample_matrices[-1].shape[0] * self.filters_enc[-1]

        self._mode = mode
        self._is_variational = is_variational
        
        self.A_edge_index, self.A_norm = self._build_adj_matrix()

        #if self.filters_enc is None:
        #    self.filters_enc = self.filters

        #if self.filters_c is None:
        #    self.filters_c = self.filters_enc

        #if self.filters_s is None:
        #    self.filters_s = self.filters_enc

        self.cheb_enc = self._build_encoder(self.filters_enc, self.K)
        self.cheb_dec_c = self._build_decoder(self.filters_dec_c, self.K)
        self.cheb_dec_s = self._build_decoder(self.filters_dec_s, self.K)
        self.pool = Pool()
        
        # Fully connected layers connecting the last pooling layer and the latent space layer.
        self.enc_lin_mu = torch.nn.Linear(self._n_features_before_z, self.z_c + self.z_s)
        
        if self._is_variational:
            self.enc_lin_var = torch.nn.Linear(self._n_features_before_z, self.z_c + self.z_s)

        if self.phase_input:
            self.dec_lin_c = torch.nn.Linear(
                self.z_c,
                self.filters_dec_c[-1]*self.upsample_matrices[-1].shape[1]
            )
            self.dec_lin_s = torch.nn.Linear(
                self.z_c + 2 * self.z_s,
                self.filters_dec_s[-1]*self.upsample_matrices[-1].shape[1]
            )
    
        self.phase_tensor = PhaseTensor()
        self.z_aggr_function = AggregatorFunction(z_aggr_function)
        self.reset_parameters()


    def _build_encoder(self, n_filters, K):
        # Chebyshev convolutions (encoder)
        cheb_enc = torch.nn.ModuleList([ ChebConv_Coma(   2*n_filters[0],  n_filters[1],  K[0])])
        cheb_enc.extend([
            ChebConv_Coma(
                n_filters[i],
                n_filters[i+1],
                K[i]
            ) for i in range(1, len(n_filters)-1)
        ])
        return cheb_enc

    def _build_decoder(self, n_filters, K):
        # Chebyshev deconvolutions (decoder)
        cheb_dec = torch.nn.ModuleList([
            ChebConv_Coma(
                n_filters[-i-1],
                n_filters[-i-2],
                K[i]
            ) for i in range(len(n_filters)-1)
        ])
        cheb_dec[-1].bias = None  # No bias for last convolution layer
        return cheb_dec

    def _build_adj_matrix(self):
        adj_edge_index, adj_norm = zip(*[
            ChebConv_Coma.norm(self.adjacency_matrices[i]._indices(), self.n_nodes[i])
            for i in range(len(self.n_nodes))
        ])
        return list(adj_edge_index), list(adj_norm)


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
            torch.nn.init.normal_(self.enc_lin_mu.weight, 0, 0.1)
        torch.nn.init.normal_(self.dec_lin_c.weight, 0, 0.1)
        torch.nn.init.normal_(self.dec_lin_s.weight, 0, 0.1)

    def encoder(self, x):
    
        for i in range(self.n_layers):
            x = self.cheb_enc[i](x, self.A_edge_index[i], self.A_norm[i])
            x = F.relu(x)
            x = self.pool(x, self.downsample_matrices[i])
    
        self.n_timeframes = x.shape[1]
        x = x.reshape(x.shape[0], self.n_timeframes, self._n_features_before_z)
               
        mu_c, log_var_c = [], []
        mu_s, log_var_s = [], []

        # Iterate through time points
        for i in range(x.shape[1]):
            mu = self.enc_lin_mu(x[:,i,:])
            mu_c.append(mu[:,:self.z_c])
            mu_s.append(mu[:,self.z_c:])
            
            if self._is_variational and self._mode == "training":
                log_var = self.enc_lin_var(x[:,i,:])
                log_var_c.append(log_var[:,:self.z_c])
                log_var_s.append(log_var[:,self.z_c:])
        
        mu_c, mu_s = torch.cat(mu_c), torch.cat(mu_s)
        mu_c = mu_c.reshape(-1, self.n_timeframes, self.z_c)
        mu_c = self.z_aggr_function(mu_c)
        mu_s = mu_s.reshape(-1, self.n_timeframes, self.z_s)
        mu_s = self.z_aggr_function(mu_s)

        if self._is_variational and self._mode == "training":
            log_var_c = torch.cat(log_var_c)
            log_var_c = log_var_c.reshape(-1, self.n_timeframes, self.z_c)
            log_var_c = self.z_aggr_function(log_var_c)

            log_var_s = torch.cat(log_var_s)
            log_var_s = log_var_s.reshape(-1, self.n_timeframes, self.z_s)
            log_var_s = self.z_aggr_function(log_var_s)

            return mu_c, log_var_c, mu_s, log_var_s

        return mu_c, mu_s


    def decoder_c(self, z_c):
                   
        x = self.dec_lin_c(z_c)
        x = x.reshape(x.shape[0], -1, self.cheb_dec_c[0].in_channels)
        for i in range(self.n_layers-1):
            x = self.pool(x, self.upsample_matrices[-i-1])
            x = self.cheb_dec_c[i](x, self.A_edge_index[self.n_layers-i-1], self.A_norm[self.n_layers-i-1])
            x = F.relu(x)
        x = self.pool(x, self.upsample_matrices[0])
        s_avg = self.cheb_dec_c[-1](x, self.A_edge_index[-1], self.A_norm[-1])
   
        return s_avg
        #TODO: comment this
        #x = x.unsqueeze(1)       


    def decoder_s(self, z_c, z_s, n_timeframes):

        s_out = []

        phased_z_s = z_s.unsqueeze(1).repeat(1, n_timeframes, *[1 for x in z_s.shape[1:]])
        phased_z_s = self.phase_tensor(phased_z_s)

        for t in range(self.n_timeframes):
           
            z_s_t = phased_z_s[:, t, ...]

            x = self.dec_lin_s(torch.cat([z_c, z_s_t], axis=-1))
            x = F.relu(x)

            # x = x.reshape(-1, self.cheb_dec[0].in_channels)
            x = x.reshape(x.shape[0], -1, self.cheb_dec_s[0].in_channels)
            for i in range(self.n_layers-1):
                x = self.pool(x, self.upsample_matrices[-i-1])
                x = self.cheb_dec_s[i](x, self.A_edge_index[self.n_layers-i-1], self.A_norm[self.n_layers-i-1])
                x = F.relu(x)
            x = self.pool(x, self.upsample_matrices[0])
            x = self.cheb_dec_s[-1](x, self.A_edge_index[-1], self.A_norm[-1])
            
            #TODO: comment this
            x = x.unsqueeze(1)
            s_out.append(x)

        s_out = torch.cat(s_out, dim=1)
        return s_out

    def _sample(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) 

    def forward(self, x):

        batch_size = x.shape[0]
        time_frames = x.shape[1]
        
        if self.phase_input:
            x = self.phase_tensor(x)
                
        x = x.reshape(batch_size, time_frames, -1, 2*self.filters_enc[0])
        
        if self._is_variational and self._mode == "training":            
            
            self.mu_c, self.log_var_c, self.mu_s, self.log_var_s = self.encoder(x)            
            z_c = self._sample(self.mu_c, self.log_var_c)
            z_s = self._sample(self.mu_s, self.log_var_s)

            if torch.isinf(z_c).any():
                self.mu_c, self.log_var_c = self.mu_c/1000, self.log_var_c/1000
                z_c = self._sample(self.mu_c, self.log_var_c)

            if torch.isinf(z_s).any():
                self.mu_s, self.log_var_s = self.mu_s/1000, self.log_var_s/1000
                z_s = self._sample(self.mu_s, self.log_var_s)
        else:
            self.mu_c, self.mu_s = self.encoder(x)
            z_c, z_s = self.mu_c, self.mu_s

        s_avg = self.decoder_c(z_c)
        ds_t = self.decoder_s(z_c, z_s, time_frames)
        s_t = s_avg.unsqueeze(1) + ds_t

        if self._is_variational and self._mode == "training":
            return (self.mu_c, self.log_var_c, self.mu_s, self.log_var_s), s_avg, s_t
        else:
            return (self.mu_c, None, self.mu_s, None), s_avg, s_t

class AggregatorFunction(nn.Module):

    def __init__(self, f):
        super(AggregatorFunction, self).__init__()
        self.z_aggregator_function = f

    def forward(self, x):
        return self.z_aggregator_function(x)
