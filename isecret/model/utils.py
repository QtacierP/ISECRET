# -*- coding: utf-8 -*-
import torch.nn as nn
import torch

def make_norm(args):
    if args.model.norm == 'in' \
            or args.model.norm.lower() == 'instancenorm':
        use_bias = True
        norm_layer = nn.InstanceNorm2d
    elif args.model.norm == 'bn' \
            or args.model.norm.lower() == 'batchnorm':
        use_bias = False
        norm_layer = nn.BatchNorm2d
    elif args.model.norm == 'sync_bn' \
            or args.model.norm.lower() == 'groupnorm':
        use_bias = False
        norm_layer = nn.SyncBatchNorm
    elif args.model.norm == 'gn' \
            or args.model.norm.lower() == 'groupnorm':
        raise NotImplementedError()  # TODO
    else:
        raise NotImplementedError()
    return norm_layer, use_bias


def make_paddding(args):
    # Set padding
    if args.model.padding == 'zero':
        padding = nn.ZeroPad2d
    elif args.model.padding == 'reflect':
        padding = nn.ReflectionPad2d
    elif args.model.padding == 'replication':
        padding = nn.ReplicationPad2d
    else:
        raise NotImplementedError
    return padding

def check_architecture(args):
    # Judge fundamental setting
    assert args.model.n_blocks > 0
    assert args.model.n_downs > 0
    assert args.model.n_filters > 0

def print_network(net, name, verbose=False):
    """Print the total number of parameters in the network and (if verbose) network architecture
    Parameters:
        verbose (bool) -- if verbose: print the network architecture
    """
    print('---------- Networks initialized -------------')
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    if verbose:
        print(net)
    print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
