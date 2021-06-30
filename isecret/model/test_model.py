from isecret.utils.metric import PSNR, SSIM
import isecret.utils.distributed as du
from isecret.model.common import MyModel
from isecret.model.build import MODEL_REGISTRY
from isecret.model.backbone import BACKBONE_REGISTRY
from isecret.dataset.common import CustomDataSet, CustomDataSetGT
from isecret.utils.transform import UnNormalize, StringTensor
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from skimage.io import imsave
import torch
import numpy as np
import os
from tqdm import tqdm
import cv2
import torch.distributed as dist
from sklearn import preprocessing

@ MODEL_REGISTRY.register('test_model')
@ torch.no_grad()
class TestModel(MyModel):
    def __init__(self, args):
        MyModel.__init__(self, args)
        if self.args.overall.gt_dir != '' or self.args.overall.test_dir == '': # With gt directory or dataset mode
            self.pair = True 

    def build_model(self):
        self.model_names = ['b2g_gen']
        try:
            self.load_model()
        except:
            self.b2g_gen = BACKBONE_REGISTRY. \
                get(self.args.model.generator.lower())(self.args)
        self._parallel()
        if self.args.overall.resume > 0:
            self.load(self.args.overall.resume)
            print('[INFO] loading {} weights'.format(self.args.overall.resume))
        elif self.args.overall.resume == -1:
            self.load('last')
            print('[INFO] loading {} weights'.format('last'))
        else:
            try:
                self.load('best')
                print('[INFO] loading best weights')
            except:
                self.load('last')
                print('[INFO] loading last weights')
        self.b2g_gen.eval()



    def on_train_begin(self):
        raise RuntimeError('Test model is only for testing')

    def inference(self, test_item, output_dir=None, importance_dir=None):
        if isinstance(test_item, DataLoader):
            metric = self.test(test_item)
        elif isinstance(test_item, str):
            metric = self.test(self._build_dataloader(test_item, gt_dir=self.args.overall.gt_dir), output_dir=output_dir)
        else:
            raise NotImplementedError
        return metric

    def _build_dataloader(self, test_dir, gt_dir=None):
        transform = transforms.Compose([transforms.Resize((self.args.data.size, self.args.data.size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(self.args.data.mean,
                                                                   self.args.data.std)])
        if gt_dir is None or gt_dir == '':
            test_dataset = CustomDataSet(self.args, test_dir, transform, need_name=True, need_shape=True)
        else:
            test_dataset = CustomDataSetGT(self.args, main_dir=test_dir,
                                           gt_dir=gt_dir, need_name=True, need_shape=True, transform=transform)
        sampler = DistributedSampler(test_dataset) if self.args.dist.n_gpus > 1 else None
        test_dataloader = DataLoader(dataset=test_dataset,
                                     batch_size=self.args.dist.batch_size_per_gpu,
                                     num_workers=self.args.dist.num_worker,
                                     pin_memory=True,
                                     sampler=sampler)
        return test_dataloader
        
    @torch.no_grad()
    def test(self, val_dataloader, output_dir=None):
        self.eval()
        metric = {}
        for i, batch in tqdm(enumerate(val_dataloader)):
            s, t, name, shape = self.test_on_batch(batch)
            [s, t] = self.recover_data([s, t])
            if output_dir is not None:
                self._save_enhancement(s, name, shape, output_dir)
            metric = self.evaluate_on_batch(s, t, metric)
        for key in metric:
            metric[key] = np.mean(metric[key])
        return metric

    def test_on_batch(self, batch):
        if 'image' in batch.keys():
            real_bad = batch['image'].to(self.args.overall.device)
        elif 'noise_good' in batch.keys(): # Use degraded image on dataset mode
            real_bad = batch['noise_good'].to(self.args.overall.device)
        elif 'bad' in batch.keys(): # Test on directory mode
            real_bad = batch['bad'].to(self.args.overall.device)
        else:
            raise RuntimeError()
        fake_good = self.b2g_gen(real_bad).detach()
        name = None
        shape = None
        # Gather name
        if 'name' in batch.keys():
            name = batch['name']
            if self.args.dist.n_gpus > 1:
                name = du.all_gather_object(name)
        # Gather shape
        if 'shape' in batch.keys():
            shape = batch['shape'].cuda()
            if self.args.dist.n_gpus > 1:
                [shape] = du.all_gather([shape])
            shape = shape.cpu().tolist()

        
        # Do not save result under the dataset mode
        if 'noise_good' in batch.keys():
            gt = batch['good'].to(self.args.overall.device)
            if self.args.dist.n_gpus > 1:
                gt, fake_good, real_bad = du.all_gather([gt, fake_good, real_bad])  # TODO: check the GPU-memory
            return fake_good, gt, name, shape

        # Output the enhancement result under the directory mode
        if self.args.dist.n_gpus > 1:
            [real_bad, fake_good] = du.all_gather([real_bad, fake_good])  # TODO: check the GPU-memory
        return fake_good.cpu(), real_bad.cpu(), name, shape



    def _save_enhancement(self, enhanced, names, shapes, output_dir):
        if not du.is_master_proc():
            return
        if not os.path.exists(output_dir):
            print('[INFO] making output dir...')
            os.makedirs(output_dir)
        for idx in range(enhanced.shape[0]):
            print(names[idx])
            _, name = os.path.split(names[idx])
            sample = enhanced[idx]
            shape = shapes[idx]
            sample = sample.permute(1, 2, 0).cpu().numpy() * 255
            sample = sample.astype(np.uint8)
            sample = cv2.resize(sample, (shape[1], shape[0]))
            imsave(os.path.join(output_dir, name), sample)







