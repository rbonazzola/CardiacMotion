import torch
from torch import nn
from torch.fft import rfft

class Mean_Aggregator(nn.Module):

    def forward(self, x):
        return torch.Tensor.mean(x, axis=1)


class FCN_Aggregator(nn.Module):

    def __init__(self, features_in, features_out):
        super(FCN_Aggregator, self).__init__()
        self.fcn = torch.nn.Linear(features_in, features_out)

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        return self.fcn(x)


class DFT_Aggregator(nn.Module):

    '''
      x [N, T, ..., F] -> [N, ..., n_comps * F]
    '''

    def __init__(self, features_in, features_out):
        super(DFT_Aggregator, self).__init__()
        self.fcn = torch.nn.Linear(features_in, features_out)

    def forward(self, x):

        x = rfft(x, dim=1)
        # Concatenate features in the frequency domain
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        x = torch.cat((x.real, x.imag), dim=-1)
        x = self.fcn(x)
        return x