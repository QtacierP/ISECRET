# -*- coding: utf-8 -*-
from torch import optim
from isecret.utils.registry import Registry
from adabelief_pytorch import AdaBelief



OPTIMIZER_REGISTRY = Registry('OPTIMIZER')
OPTIMIZER_REGISTRY.__doc__ = 'Optimizer for building model'

OPTIMIZER_REGISTRY.register('sgd', optim.SGD)
OPTIMIZER_REGISTRY.register('adam', optim.Adam)
OPTIMIZER_REGISTRY.register('asgd', optim.ASGD)


def make_optimizer(args, parameters):
    if args.train.optim.lower() == 'sgd':
        return optim.SGD(params=parameters, lr=args.train.lr,
                         momentum=args.train.momentum, weight_decay=args.train.weight_decay)
    elif args.train.optim.lower() == 'adam':
        return optim.Adam(params=parameters, lr=args.train.lr,
                          betas=(args.train.beta1, args.train.beta2),weight_decay=args.train.weight_decay)
    elif args.train.optim.lower() == 'adam_belief':
        return AdaBelief(params=parameters, lr=args.train.lr,
                         betas=(args.train.beta1, args.train.beta2), weight_decay=args.train.weight_decay)
    else:
        raise NotImplementedError('{} optimizer is not supported !'.format(args.train.optim))


