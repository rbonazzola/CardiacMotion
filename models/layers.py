import itertools
import torch
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.cheb_conv import ChebConv
from torch_geometric.utils import remove_self_loops

from subprocess import check_output
import shlex
# import sys; sys.path.append(".")
repo_root = check_output(shlex.split("git rev-parse --show-toplevel")).strip().decode('ascii')

from IPython import embed
from utils.utils import normal

__author__ = ['Priyanka Patel', 'Rodrigo Bonazzola']

# N: number of subjects (or, equivalently, meshes)
# M: number of vertices in mesh
# F: number of features (typically, 3: x, y and z)

# https://pytorch-geometric.readthedocs.io/en/1.3.1/_modules/torch_geometric/nn/conv/cheb_conv.html
# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/cheb_conv.html#ChebConv
class ChebConv_Coma(ChebConv):

    def __init__(self, in_channels, out_channels, K, normalization=None, bias=True):
        super(ChebConv_Coma, self).__init__(in_channels, out_channels, K, normalization, bias)

   #def reset_parameters(self):
   #     embed()
   #     normal(self.weight, 0, 0.1) # Same as torch.nn.init.normal_ but handles None's
   #     normal(self.bias, 0, 0.1)


    # Normalized Laplacian. This is almost entirely copied from the parent class, ChebConv.
    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, dtype=None):
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes) # TODO: Check what scatter_add does.
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return edge_index, -deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


    def forward(self, x, edge_index, norm, edge_weight=None):
        # Tx_i are Chebyshev polynomials of x, which are computed recursively
        Tx_0 = x # Tx_0 is the identity, i.e. Tx_0(x) == x

        #TOFIX: This is a workaround to make my code work with a newer version of PyTorch (1.10),
        #since the weight attribute seems to be absent in this version.
        self.weight = []
        #TODO: change this range
        for i in range(1, 7):            
            try:
              self.weight.append(next(itertools.islice(self.parameters(), i, None)).t())
            except:
              pass
        
        out = torch.matmul(Tx_0, self.weight[0])

        # if self.weight.size(0) > 1:
        if len(self.weight) > 1:
            Tx_1 = self.propagate(edge_index, x=Tx_0, norm=norm) # propagate amounts to operator composition
            out = out + torch.matmul(Tx_1, self.weight[1])

        # for k in range(2, self.weight.size(0)):
        for k in range(2, len(self.weight)):
            Tx_2 = 2 * self.propagate(edge_index, x=Tx_1, norm=norm) - Tx_0 # recursive definition of Chebyshev polynomials
            out = out + torch.matmul(Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j, norm):
        # x_j is in the format (N, M, F)
        return norm.view(1, -1, 1) * x_j


class Pool(MessagePassing):
    def __init__(self):
        # source_to_target is the default value for flow, but is specified here for explicitness
        super(Pool, self).__init__(flow='source_to_target')

    def forward(self, x, pool_mat,  dtype=None):
        pool_mat = pool_mat.transpose(0, 1)
        out = self.propagate(edge_index=pool_mat._indices(), x=x, norm=pool_mat._values(), size=pool_mat.size())
        return out

    def message(self, x_j, norm):
        return norm.view(1, -1, 1) * x_j




class PhaseTensor1(nn.Module):

    def phase_tensor(self, z, n_timeframes):
        '''
        params:
          z: a batched vector (N x M)

        returns:
          a phase-aware vector (N x T x 2M)
        '''


        exp_it = []

        for i in range(n_timeframes):
            phase = 2 * np.pi * i / n_timeframes
            exp_it.append([np.sin(phase), np.cos(phase)])

        exp_it = np.array(exp_it)
        exp_it = np.expand_dims(exp_it, axis=(0, 2))
        exp_it = torch.Tensor(exp_it)
        exp_it = exp_it.reshape(n_timeframes, 1, 2)
        exp_it = exp_it.type_as(z)

        # trick to achieve the desired dimensions
        z = z.reshape(-1, 1, self.z, 1)
        z_t = torch.matmul(z, exp_it)

        z_t = z_t.reshape(*z_t.shape[:2], -1)

        return z_t

    def forward(self, z):
        return self.phase_tensor(z)


class PhaseTensor2(nn.Module):

    def phase_tensor(self, x):
        '''
        params:
            x: a real-valued Tensor of dimensions [batch_size, n_phases, ...]

        return:
            real-valued Tensor of twice the last dimension as the input
        '''

        # x.shape[1] is the number of phases

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