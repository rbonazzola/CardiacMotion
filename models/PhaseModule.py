import torch
import numpy as np
from torch import nn

class PhaseTensor(nn.Module):

    def __init__(self, version="version_1"):

        super(PhaseTensor, self).__init__()
        self.version = version

    def phase_tensor(self, x, ):
        '''
        params:
          z: a batched vector (N x T x M)

        returns:
          a phase-aware vector (N x T x 2M)
        '''

        if self.version == "version_1":

            sen_t = []; cos_t = []
            n_timeframes, rank = x.shape[1], x.dim()

            for i in range(n_timeframes):
                phase = 2 * np.pi * i / n_timeframes
                sen_t.append(np.sin(phase))
                cos_t.append(np.cos(phase))

            dims_to_expand = list(range(rank))
            dims_to_expand.remove(1)  # don't expand along the "time" dimension
            dims_to_expand = tuple(dims_to_expand)

            sen_t = np.array(sen_t); cos_t = np.array(cos_t)
            sen_t = np.expand_dims(sen_t, axis=dims_to_expand)
            cos_t = np.expand_dims(cos_t, axis=dims_to_expand)
            sen_t = torch.Tensor(sen_t); cos_t = torch.Tensor(cos_t)
            # sen_t = sen_t.type_as(x); cos_t = cos_t.type_as(x)

            phased_x = torch.cat((sen_t * x, cos_t * x), dim=-1)
            phased_x.type_as(x)

        elif self.version == "version_2":

            phased_x = x.type(torch.complex64)
            n_timeframes = x.shape[1]

            for t in range(n_timeframes):
                phase = 2 * np.pi * t / n_timeframes * torch.ones_like(x[:, t, ...])
                phase = torch.FloatTensor(phase)
                phase = phase.type_as(x)  # to(x.device)

                # torch.polar(x, phase) returns x * exp(i * phase), i.e. x as a phasor
                phased_x[:, t, ...] = torch.polar(x[:, t, ...], phase)

            # concatenate sin and cosine along last dimension
            phased_x = torch.cat((phased_x.real, phased_x.imag), dim=-1)

        return phased_x

    def forward(self, x):
        return self.phase_tensor(x)