# -*- coding: utf-8 -*-
from isecret.utils.registry import Registry
from torch.optim import lr_scheduler


SCHEDULER_REGISTRY = Registry('SCHEDULER')
SCHEDULER_REGISTRY.__doc__ = 'Learning rate scheduler for building model'

SCHEDULER_REGISTRY.register('cosine', lr_scheduler.CosineAnnealingLR)
SCHEDULER_REGISTRY.register('plateau', lr_scheduler.ReduceLROnPlateau)
SCHEDULER_REGISTRY.register('exp', lr_scheduler.ExponentialLR)
SCHEDULER_REGISTRY.register('multiple', lr_scheduler.MultiStepLR)
SCHEDULER_REGISTRY.register('none', None)