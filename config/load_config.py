import yaml
from argparse import Namespace

def recursive_namespace(dd):
    '''
    converts a (possibly nested) dictionary into a namespace
    '''
    for d in dd:
        has_any_dicts = False
        if isinstance(dd[d], dict):
            dd[d] = recursive_namespace(dd[d])
            has_any_dicts = True
    return Namespace(**dd)

#from utils.generics import recursive_namespace

def load_config(yaml_config_file):
    
    
    with open(yaml_config_file) as config:
        config = yaml.safe_load(config)    
        # I am using a namespace instead of a dictionary mainly because it enables auto-completion
        config = recursive_namespace(config)
    
    config.network_architecture.convolution.parameters.polynomial_degree = \
    [int(x) for x in config.network_architecture.convolution.parameters.polynomial_degree.split()]
    
    config.network_architecture.pooling.parameters.downsampling_factors = \
    [int(x) for x in config.network_architecture.pooling.parameters.downsampling_factors.split()]
    
    config.network_architecture.convolution.parameters.channels = \
    [int(x) for x in config.network_architecture.convolution.parameters.channels.split()]
    
    return config
