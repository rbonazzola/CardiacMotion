optimizers_menu = {
  "adam": torch.optim.Adam(coma.parameters(), lr=lr, betas=(0.5,0.99), weight_decay=weight_decay),
  "sgd": torch.optim.SGD(coma.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
}

losses_menu = {
  "l1": {"name": "L1", "function": F.l1_loss},
  "mse": {"name": "MSE", "function": F.mse_loss}
}


if __name__ == '__main__':

    import argparse
    
    def overwrite_config_items(config, args):
      for attr, value in args.__dict__.items():
        if attr in config.keys() and value is not None:
          config[attr] = value
       
    parser = argparse.ArgumentParser(description='Pytorch Trainer for Convolutional Mesh Autoencoders')

    parser.add_argument('-c', '--conf', help='path of config file', default="config_files/default.cfg")
    parser.add_argument('-od', '--output_dir', default=None, help='path where to store output')
    parser.add_argument('-id', '--data_dir', default=None, help='path where to fetch input data from')
    parser.add_argument('--preprocessed_data', default=None, type=str, help='Location of cached input data.')
    parser.add_argument('--partition', default=None, type=str, help='Cardiac chamber.')
    parser.add_argument('--procrustes_scaling', default=False, action="store_true", help="Whether to perform scaling transformation after Procrustes alignment (to make mean distance to origin equal to 1).")
    parser.add_argument('--phase', default=None, help="cardiac phase (1-50|ED|ES)")
    parser.add_argument('--z', default=None, type=int, help='Number of latent variables.')
    parser.add_argument('--optimizer', default=None, type=str, help='optimizer (adam or sgd).')
    parser.add_argument('--epoch', default=None, type=int, help='Maximum number of epochs.')
    parser.add_argument('--nTraining', default=None, type=int, help='Number of training samples.')
    parser.add_argument('--nVal', default=None, type=int, help='Number of validation samples.')
    parser.add_argument('--kld_weight', type=float, default=None, help='Weight of Kullback-Leibler divergence.')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate.')
    parser.add_argument('--seed', default=None, help="Seed for PyTorch's Random Number Generator.")
    parser.add_argument('--stop_if_not_learning', default=None, action="store_true", help='Stop training if losses do not change.')
    parser.add_argument('--save_all_models', default=False, action="store_true",
                        help='Save all models instead of just the best one until the current epoch.')
    parser.add_argument('--test', default=False, action="store_true", help='Set this flag if you just want to test whether the code executes properly. ')
    parser.add_argument('--dry-run', dest="dry_run", default=False, action="store_true",
                        help='Dry run: just prints out the parameters of the execution but performs no training.')

    args = parser.parse_args()

    if args.conf is None:
        args.conf = os.path.join(os.path.dirname(__file__), 'default.cfg')
        logger.error('configuration file not specified, trying to load '
              'it from current directory', args.conf)

    ################################################################################
    ### Load configuration
    if not os.path.exists(args.conf):
        logger.error('Config not found' + args.conf)
    config = read_config(args.conf)

    if args.data_dir:
        config['data_dir'] = args.data_dir

    if args.output_dir:
        config['output_dir'] = args.output_dir

    if args.test:
        # some small values so that the execution ends quickly
        config['comments'] = "this is a test"
        config['nTraining'] = 500
        config['nVal'] = 80
        config['epoch'] = 20
        config['output_dir'] = "output/test_{TIMESTAMP}"
    config['test'] = args.test
       
    overwrite_config_items(config, args)

    if args.dry_run:
      pprint(config)
      exit()

    main(config)

