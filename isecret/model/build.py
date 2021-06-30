# -*- coding: utf-8 -*-
from isecret.utils.registry import Registry

MODEL_REGISTRY = Registry('MODEL')
MODEL_REGISTRY.__doc__ = 'Model registry'


def build_model(args, gpu=None):
    name = args.model.model.lower()
    model = MODEL_REGISTRY.get(name)(args)
    return model