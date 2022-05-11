class Coma4D_C_and_S(torch.nn.Module):

    def __init__(self,
          num_features,
          num_conv_filters_enc,
          polygon_order,
          latent_dim_content,
          latent_dim_style,
          is_variational,
          downsample_matrices,
          upsample_matrices,
          adjacency_matrices,
          template,
          n_nodes,
          z_aggr_function,
          phase_input,
          n_timeframes=None,
          mode="testing"
    	):

        self.encoder = Encoder(
            n_layers,
            phase_input,
            num_conv_filters_enc,
            num_features,
            adjacency_matrices,
            polygon_order,
            is_variational,
            latent_dim_content,
            latent_dim_style,
            downsample_matrices,
            z_aggr_function
        )

        self.decoder = Decoder(
            latent_dim_content,
            latent_dim_style,
            n_layers,
            num_conv_filters_dec_c,
            num_conv_filters_dec_s,
            upsample_matrices,
            adjacency_matrices
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat



class EncoderTemporalSequence(nn.Module):


    def __init__(self, encoder_config, z_aggr_function):

        self.encoder_3d_mesh = Encoder3DMesh(**encoder_config)
        self.z_aggr_function = self._get_z_aggr_function(z_aggr_function)


    def _get_z_aggr_function(self, z_aggr_function):

        if z_aggr_function == "mean":
            z_aggr_function Mean_Aggregator()

        elif z_aggr_function.lower() == "fcn" or z_aggr_function.lower() == "fully_connected":
            self.n_timeframes = n_timeframes
            z_aggr_function = FCN_Aggregator(
                features_in=n_timeframes * (self.z_c + self.z_s),
                features_out=(self.z_c + self.z_s)
            )

        elif z_aggr_function.lower() == "dft" or z_aggr_function.lower() == "discrete_fourier_transform":
            self.n_timeframes = n_timeframes
            z_aggr_function = DFT_Aggregator(
                features_in=(n_timeframes // 2 + 1) * 2 * (self.z_c + self.z_s),
                features_out=(self.z_c + self.z_s)
            )

        return z_aggr_function

    def encoder(self, x):

        self.n_timeframes = x.shape[1]

        x = self.encoder_3d_mesh(x)
        x = self.concatenate_graph_features(x)

        mu, log_var = [], []

        # Iterate through time points
        for i in range(self.n_timeframes):
            _mu = self.enc_lin_mu(x[:, i, :])
            mu.append(_mu)

            if self._is_variational and self._mode == "training":
                log_var = self.enc_lin_var(x[:, i, :])
                log_var.append(_log_var)
            else:
                log_var = None

        mu = torch.cat(mu).reshape(-1, self.n_timeframes, self.z_c + self.z_s)
        mu = self.z_aggr_function(mu)

        if log_var is not None:
            log_var_t = torch.cat(log_var).reshape(-1, self.n_timeframes, self.z_c + self.z_s)
            log_var = self.z_aggr_function(log_var_t)

        bottleneck =  self._partition_z(mu, log_var)
        return bottleneck

    def _partition_z(self, mu, log_var=None):

        mu_c = mu[:, :self.z_c]
        mu_s = mu[:, self.z_c:]
        bottleneck = {"mu_s": mu_s, "mu_c": mu_c}

        if log_var is not None:
            log_var_c = log_var[:, :self.z_c]
            log_var_s = log_var[:, self.z_c:]
            bottleneck["log_var_c"] = log_var_c
            bottleneck["log_var_s"] = log_var_s

        return bottleneck


class Encoder3DMesh(nn.Module):

    '''
    '''

    def __init__(self, **kwargs):

        self.phase_input = phase_input
        self.filters_enc = num_conv_filters_enc
        self.filters_enc.insert(0, num_features)
        self.K = polygon_order
        self.adjacency_matrices = adjacency_matrices

        self._is_variational = is_variational

        self.z_c = latent_dim_content
        self.z_s = latent_dim_style

        self.downsample_matrices = downsample_matrices
        self.A_edge_index, self.A_norm = self._build_adj_matrix()
        self.cheb_enc = self._build_cheb_conv_layers(self.filters_enc, self.K)
        self.pool = Pool()
        self.activation_layers = [activation_layers] * n_layers if isinstance(activation_layers, str) else activation_layers

        # Fully connected layers connecting the last pooling layer and the latent space layer.
        self.enc_lin_mu = torch.nn.Linear(self._n_features_before_z, self.z_c + self.z_s)

        if self._is_variational:
            self.enc_lin_var = torch.nn.Linear(self._n_features_before_z, self.z_c + self.z_s)

    def _build_encoder(self):

        cheb_conv_layers = self._build_cheb_conv_layers(self.n_filters, self.K)
        pool_layers = self._build_pool_layers(self.downsample_matrices)
        activation_layers = self._build_activation_layers()

        encoder = ModuleList()
        for i in range(len(cheb_conv_layers)):
            encoder.append(cheb_conv_layers[i])
            encoder.append(pool_layers[i])
            encoder.append(activation_layers[i])
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
            activation_layers.append(F._getattr(activation_type[i]))
        return activation_layers


    def _build_cheb_conv_layers(self, n_filters, K):
        # Chebyshev convolutions (encoder)
        if self.phase_input:
            cheb_enc = torch.nn.ModuleList([ChebConv_Coma(2 * n_filters[0], n_filters[1], K[0])])
        else:
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