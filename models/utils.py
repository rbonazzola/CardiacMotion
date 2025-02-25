import torch

def normal(tensor, mean, std):
    if tensor is not None:
        torch.nn.init.normal_(tensor, mean=mean, std=std)