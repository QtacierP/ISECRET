# -*- coding: utf-8 -*-
from isecret.utils.registry import Registry
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

DATASET_REGISTRY = Registry('DATASET')
DATASET_REGISTRY.__doc__ = 'Dataset for fundus-enhancement'


def build_dataset(args, task):
    name = args.data.dataset.lower()
    return DATASET_REGISTRY.get(name)(args, task)


def build_loader(args):
    tasks = ['train', 'val', 'test']
    data_loaders = []

    for task in tasks:
        #print(task)
        dataset = build_dataset(args, task)
        sampler = DistributedSampler(dataset) if args.dist.n_gpus > 1 else None
        shuffle = True if sampler is None else False
        data_loaders.append(
            DataLoader(
                dataset=dataset,
                batch_size=args.dist.batch_size_per_gpu,
                num_workers=args.dist.num_worker,
                pin_memory=True,
                sampler=sampler,
                shuffle=shuffle
            )
        )
    return tuple(data_loaders)