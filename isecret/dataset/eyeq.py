# -*- coding: utf-8 -*-
from isecret.dataset.common import BaseDataset, CustomDataSet, OfflineDegradeCustomDataSet
from isecret.dataset.build import DATASET_REGISTRY
import os
from torchvision import transforms


@DATASET_REGISTRY.register('eyeq')
class Eyeq(BaseDataset):
    '''
    EyeQ Dataset, which is an unpaired dataset
    '''
    def __init__(self, args, task):
        self.task = task
        self.args = args
        self._init_dataset()
        BaseDataset.__init__(self, args, self.good_dataset, self.bad_dataset)

    def _init_dataset(self):
        # Initialize pre-processing
        if self.task == 'train':
            transform = transforms.Compose([transforms.Resize((self.args.data.size, self.args.data.size)),
                                            transforms.RandomCrop((self.args.data.crop_size, self.args.data.crop_size)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomVerticalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(self.args.data.mean,
                                                                 self.args.data.std)])
        else:
            transform = transforms.Compose([transforms.Resize((self.args.data.size, self.args.data.size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(self.args.data.mean,
                                                                 self.args.data.std)])
        # Initialize directory
        dataset = OfflineDegradeCustomDataSet
        good_dir = os.path.join(self.args.data.data_dir, self.task, 'crop_good')
        degrade_dir = os.path.join(self.args.data.data_dir, self.task, 'degrade_good')
        self.good_dataset = dataset(self.args, good_dir, degrade_dir, transform)
        bad_dir = os.path.join(self.args.data.data_dir, self.task, 'crop_usable')
        self.bad_dataset = CustomDataSet(self.args, bad_dir, transform)



@ DATASET_REGISTRY.register('minieyeq')
class MiniEyeq(Eyeq):
    def __init__(self, args, task):
        super().__init__(args, task)