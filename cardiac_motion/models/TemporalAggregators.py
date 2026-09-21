import math
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

    
class _RoPESelfAttention(nn.Module):

    '''
    Multi-head self-attention with rotary position embeddings (RoPE) applied
    to queries/keys, indexed by *cardiac phase* (2*pi*t/T) rather than raw
    frame index -- consistent with the phase convention already used for the
    style decoder (see PhaseModule.PhaseTensor). Written by hand (instead of
    nn.MultiheadAttention) so RoPE can be injected into Q/K before the dot
    product.
    '''

    def __init__(self, d_model, n_heads, base=10000.0):

        super(_RoPESelfAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rope_cos_sin(self, phase):
        # phase: [T] (radians) -> cos, sin: [T, head_dim]
        freqs = phase[:, None] * self.inv_freq[None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x, phase):
        # x: [N, T, d_model], phase: [T] (radians)
        N, T, D = x.shape
        qkv = self.qkv(x).reshape(N, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [N, n_heads, T, head_dim]

        cos, sin = self._rope_cos_sin(phase.to(x.device))
        cos, sin = cos[None, None], sin[None, None]  # [1, 1, T, head_dim]
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin

        attn = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(N, T, D)
        return self.out_proj(out)


class _TransformerAggregatorLayer(nn.Module):

    '''Pre-norm transformer encoder block: RoPE self-attention + GELU FFN.'''

    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):

        super(_TransformerAggregatorLayer, self).__init__()
        self.attn = _RoPESelfAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, phase):
        x = x + self.dropout(self.attn(self.norm1(x), phase))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class TransformerAggregator(nn.Module):

    '''
    Per-frame spatial features [N, T, F_spatial] -> linear projection to
    d_model -> a few RoPE-transformer encoder layers (self-attention over
    the T frames, positioned by cardiac phase rather than raw index, so the
    encoding is consistent with the periodic 0..2*pi convention already used
    for the style decoder) -> mean pool over time -> linear to latent_dim.

    Unlike FCN_Aggregator, parameter count does not depend on n_timeframes
    (only on F_spatial, d_model, and the number/width of layers), so the
    same weights generalize across different T.
    '''

    def __init__(self, features_in, features_out, n_timeframes=None,
                 d_model=128, n_heads=4, n_layers=2, d_ff=256, dropout=0.0):

        super(TransformerAggregator, self).__init__()
        self.n_timeframes = n_timeframes

        self.proj_in = nn.Linear(features_in, d_model)
        self.layers = nn.ModuleList([
            _TransformerAggregatorLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)
        # named "fcn" (not "proj_out") so EncoderTemporalSequence's near-zero
        # init of the final projection layer (`.fcn.weight`) keeps working
        # unmodified for this aggregator too.
        self.fcn = nn.Linear(d_model, features_out)

    @staticmethod
    def _phase(n_timeframes, device):
        return 2 * math.pi * torch.arange(n_timeframes, device=device, dtype=torch.float32) / n_timeframes

    def forward(self, x):
        # x: [N, T, F_spatial]
        N, T, F = x.shape
        phase = self._phase(T, x.device)

        h = self.proj_in(x)
        for layer in self.layers:
            h = layer(h, phase)
        h = self.norm_out(h)
        h = h.mean(dim=1)  # mean pool over time
        return self.fcn(h)


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