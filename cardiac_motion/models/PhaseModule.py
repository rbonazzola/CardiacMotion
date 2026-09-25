import torch
import numpy as np
from torch import nn

class PhaseTensor(nn.Module):

    def __init__(self, version="version_1", n_harmonics=1):

        super(PhaseTensor, self).__init__()
        self.version = version
        self.n_harmonics = n_harmonics
        if n_harmonics < 1:
            raise ValueError(f"n_harmonics must be >= 1, got {n_harmonics}")
        if version != "version_1" and n_harmonics != 1:
            raise NotImplementedError(f"n_harmonics > 1 is only implemented for version_1, not {version}")

    def phase_tensor(self, x):
        '''
        params:
          z: a batched vector (N x T x M)

        returns:
          a phase-aware vector (N x T x 2KM), K = n_harmonics:
          [sin(theta) x, cos(theta) x, sin(2 theta) x, cos(2 theta) x, ...], theta = 2 pi t / T.
          For K = 1 this is the original [sin(theta) x, cos(theta) x].
        '''

        if self.version == "version_1":

            n_timeframes, rank = x.shape[1], x.dim()

            dims_to_expand = list(range(rank))
            dims_to_expand.remove(1)  # don't expand along the "time" dimension
            dims_to_expand = tuple(dims_to_expand)

            parts = []
            for k in range(1, self.n_harmonics + 1):
                phase = 2 * np.pi * k * np.arange(n_timeframes) / n_timeframes
                for trig in (np.sin, np.cos):
                    wave = torch.Tensor(np.expand_dims(trig(phase), axis=dims_to_expand)).type_as(x)
                    parts.append(wave * x)

            phased_x = torch.cat(parts, dim=-1)

        elif self.version == "version_2":

            phased_x = x.type(torch.complex64)
            n_timeframes = x.shape[1]

            for t in range(n_timeframes):
                phase = 2 * np.pi * t / n_timeframes * torch.ones_like(x[:, t, ...])
                # phase = torch.FloatTensor(phase)
                phase = phase.type_as(x)  # to(x.device)

                # torch.polar(x, phase) returns x * exp(i * phase), i.e. x as a phasor
                phased_x[:, t, ...] = torch.polar(x[:, t, ...], phase)

            # concatenate sin and cosine along last dimension
            phased_x = torch.cat((phased_x.real, phased_x.imag), dim=-1)

        return phased_x

    def forward(self, x):
        return self.phase_tensor(x)
