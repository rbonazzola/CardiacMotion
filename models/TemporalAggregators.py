import torch
from torch import nn
from torch.fft import rfft

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