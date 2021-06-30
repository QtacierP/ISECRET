# -*- coding: utf-8 -*-
from torch import distributed
from isecret.model.utils import  print_network
from isecret.utils.transform import UnNormalize
from isecret.train_utils.utils import TorchBar
from isecret.train_utils.optimizer import make_optimizer
from isecret.train_utils.scheduler import SCHEDULER_REGISTRY
import isecret.utils.distributed as du
from isecret.utils.metric import PSNR, SSIM
import isecret.utils.io as io
from abc import ABC
import torch.utils.tensorboard as tsb
import torch
from torchvision.utils import make_grid
import numpy as np
import os
from itertools import chain
import pickle
import time


class MyModel(ABC):
    def __init__(self, args):
        self.args = args
        try:
            if self.args.train.default:
                self._modify_args(args) 
        except:
            self.args = args
        self._check_network()
        self.model_names = None
        # TODO: Check with cpu setting
        self.master = du.is_master_proc\
            (args.dist.n_gpus * args.dist.num_shards) or self.args.dist.n_gpus == 1 # is master process or not
        self.dist = args.dist.n_gpus > 1
        self.pair = False
        self.build_model()

    def _check_network(self):
        # Check network
        pass

    def _modify_args(self):
        # Change configs 
        pass


    '''
    ==========================
    === Build architecture ===
    ==========================
    '''
    def build_model(self):
        # Build your model architecture here
        pass

    def _parallel(self, model_names=None):
        print('[INFO] Parallelize model...')
        if self.args.overall.device == 'cpu':
            return  # TODO distributed training with cpu
        if model_names is None:
            model_names = self.model_names
        for model_name in model_names:
            model = getattr(self, model_name)
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                continue  # TODO: check the instance
            gpu = torch.cuda.current_device()
            model = model.cuda(device=gpu)
            if self.args.dist.n_gpus > 1:
                model = torch.nn.parallel.DistributedDataParallel(
                    module=model, device_ids=[gpu], output_device=gpu)
                setattr(self, model_name, model)


    def build_loss(self):
        # Build your loss function here
        print('[INFO] Building loss...')


    def build_optimizer(self):
        print('[INFO] Building optimizer ...')
        # Build your optimizer here
        self.gen_optimizer = make_optimizer(self.args,
                                            chain(*[getattr(self, name).parameters()
                                                    for name in self.model_names if 'gen' in name]))
        if self.args.train.lambda_gan > 0:
            self.dis_optimizer = make_optimizer(self.args,
                                                chain(*[getattr(self, name).parameters()
                                                        for name in self.model_names if 'dis' in name]))
        self.optim_dict = {}
        for model_name in self.model_names:
            if 'gen' in model_name:
                self.optim_dict[model_name] = self.gen_optimizer
            elif 'dis' in model_name:
                self.optim_dict[model_name] = self.dis_optimizer

    def build_scheduler(self):
        print('[INFO] Building scheduler ...')
        # Build your lr_scheduler here
        self.gen_scheduler = SCHEDULER_REGISTRY.get(self.args.train.scheduler) \
            (self.gen_optimizer, T_max=self.args.train.epochs)
        if self.args.train.lambda_gan > 0:
            self.dis_scheduler = SCHEDULER_REGISTRY.get(self.args.train.scheduler) \
                (self.dis_optimizer, T_max=self.args.train.epochs)
        self.scheduler_dict = {}
        for model_name in self.model_names:
            if 'gen' in model_name:
                self.scheduler_dict[model_name] = self.gen_scheduler
            elif 'dis' in model_name:
                self.scheduler_dict[model_name] = self.dis_scheduler

    def build_logger(self):
        if not self.master:
            return
        self.writer = tsb.SummaryWriter(self.args.experiment.ckpt_dir)

    '''
    ===========================
    ======= Training log ======
    ===========================
    '''

    def on_train_begin(self):
        # Build fundamental items
        self.build_loss()
        self.build_optimizer()
        self.build_scheduler()
        self.build_logger()

        # Print network
        for name in self.model_names:
            print_network(getattr(self, name), name)

        if 'loss' in self.args.train.metric:
            self.opt = np.less
            self.best = np.inf
        else:
            self.opt = np.greater
            self.best = -np.inf
        # Initialize training setting
        self.epoch = 1
        self.start_epoch = 1
        self.load_or_not()  # Update epoch & start epoch
        if not self.master:
            return
        if self.epoch == 1:
            self.save_model()
        self.batch_num = 0
        self.global_step = self.step * (self.epoch - 1) + self.batch_num

    def on_epoch_begin(self):
        # Set zero for loss and batch_num
        self.losses = {}
        self.batch_num = 0
        if not self.master:
            return
        print('\n===> Epoch {} <==='.format(self.epoch))
        self.bar = TorchBar(target=self.step, width=30)
        print('Current learning rate is {}'.format(self.gen_optimizer.param_groups[0]['lr']))

    def _adjust_lr(self):
        # Adjust learning rate
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()
            self.dis_scheduler.step()

  
    def on_batch_begin(self):
        pass
    
    @ torch.no_grad()
    def on_batch_end(self, losses: dict, meta=None):
        '''
        Record loss and visualize data on tensorboard
        :param losses: losses to store
        :param meta: visualized data
        :return:
        '''
        self.batch_num += 1
        if not self.master:
            return
        # Log loss
        values = []
        for loss_name in losses.keys():
            # TODO: Abandon the cpu loss
            if isinstance(losses[loss_name], int) or isinstance(losses[loss_name],
                                                                float):  # ignore the meaningless loss func
                loss = losses[loss_name]
            else:
                loss = losses[loss_name].detach().cpu().numpy()
            if loss == 0:
                continue
            if loss_name not in self.losses.keys():
                self.losses[loss_name] = []
            self.losses[loss_name].append(loss)
            values.append((loss_name, loss))

        # Visualize log
        self.bar.update(self.batch_num , values=values)
        self.global_step = self.step * (self.epoch - 1) + self.batch_num
        if self.global_step % self.args.train.sample_freq == 0:
            un_norm = UnNormalize(self.args.data.mean,
                                  self.args.data.std)
            if meta is not None:
                # Visualize images
                for tag in meta.keys():
                    img_list = meta[tag]
                    imgs = torch.cat([img_list], dim=0)
                    imgs = make_grid(imgs, nrow=self.args.dist.batch_size_per_gpu)
                    if 'mask' not in tag:
                        imgs = un_norm(imgs)
                    imgs = imgs.clamp(min=0.0, max=1.0)
                    self.writer.add_image(tag, imgs, global_step=self.global_step)


    def on_epoch_end(self, val_metric=None):
        '''
        Write loss/metric to tensorboard
        Save model checkpoints
        :param val_metric: metric on validation dataset
        :return:
        '''
        self._adjust_lr()
        if not self.master:
            return
        # Delete the TorchBar
        del (self.bar)
        # Write loss to log
        for loss_name in self.losses:
            loss = np.mean(self.losses[loss_name])
            self.writer.add_scalar(loss_name, loss,
                                   self.epoch)
            self.losses[loss_name] = []

        if self.epoch % self.args.train.save_freq == 0:
            self.save(flag=str(self.epoch))

        self.save(flag='last')
        self.epoch += 1
        
        if val_metric is None:
            return
        # Log validation metric
        print('\nValidation result =>')
        if self.args.train.metric not in val_metric.keys():
            print('[Warning]: Missing metric {}'.format(self.args.train.metric))
        for metric_name in val_metric.keys():
            self.writer.add_scalar(metric_name, val_metric[metric_name],
                                   self.epoch)
            print('{}: {}'.format(metric_name, val_metric[metric_name]))
            if metric_name == self.args.train.metric:
                if self.opt(val_metric[metric_name], self.best):
                    print('{} improved from {} to {}'.format(metric_name, self.best,
                                                             val_metric[metric_name]))
                    self.best = val_metric[metric_name]
                    print('\n[INFO]: saving model weights')
                    self.save()
                    # Save best metric
                    io.save_json(os.path.join(self.args.experiment.experiment_dir,
                                          'best_{}.json'.format(self.args.train.metric)), val_metric)

    def on_train_end(self):
        if not self.master:
            return

    '''
       ================================
       ====== Distributed gather ======
       ================================
    '''
    def gather_item(self, item):
        # Gather items between gpus
        if not self.dist:
            return item
        if isinstance(item, dict):
            for key in item.keys():
                item[key] = du.all_gather_unaligned(item[key])
        elif isinstance(item, list):
            item = [du.all_gather_unaligned(item)]
        else:
            item = du.all_gather_unaligned(item)
        return item

    def reduce_item(self, item):
        # Reduce item between gpus
        if not self.dist:
            return item
        if isinstance(item, dict):
            for key in item.keys():
                [item[key]] = du.all_reduce([item[key]])
        elif isinstance(item, list):
            item = du.all_reduce(item)
        else:
            [item] = du.all_reduce([item])
        return item


    '''
    ===============================
    ====== model save & load ======
    ================================
    '''
    def load_model(self):
        # load model class
        print('[INFO] Loading model arch...')
        map_location = {'cuda:%d' % 0: 'cuda:%d' % torch.cuda.current_device()}
        for model_name in self.model_names:
            f = open(os.path.join(self.args.experiment.experiment_dir, '{}.pt'.format(model_name)), 'rb')
            model = torch.load(f, map_location=map_location)
            setattr(self, model_name, model)
            f.close()

    def save_model(self):
        # Save model class
        if not self.master:
            return
        for model_name in self.model_names:
            model = getattr(self, model_name)
            if self.args.dist.n_gpus > 1:
                model = model.module
            f = open(os.path.join(self.args.experiment.experiment_dir, '{}.pt'.format(model_name)), 'wb')
            torch.save(model, f)
            f.close()

    def save(self, flag='best'):
        # save weights and optimizer
        if not self.master:
            return
        for model_name in self.model_names:
            model = getattr(self, model_name)
            if self.args.dist.n_gpus > 1:
                model = model.module
            state = {
                    'epoch': self.epoch,
                    'optimizer': self.optim_dict[model_name].state_dict(),
                    'weights': model.state_dict(),
                    'scheduler': self.scheduler_dict[model_name].state_dict()
            }
            torch.save(state, os.path.join(self.args.experiment.experiment_dir, '{}_{}.pt'.format(model_name, flag)))

    def load(self, flag='best'):
        # Load weights, optimizer and history metric
        print('\n[INFO]: laoding model weights ...')
        for model_name in self.model_names:
            state = torch.load(os.path.join(self.args.experiment.experiment_dir,
                                            '{}_{}.pt'.format(model_name, flag)), map_location=lambda storage, loc: storage)
            model = getattr(self, model_name)
            try:
                model.load_state_dict(state['weights'])
            except:
                model.module.load_state_dict(state['weights'])
            if not self.args.overall.test:
                optimizer = self.optim_dict[model_name]
                optimizer.load_state_dict(state['optimizer'])
                scheduler = self.scheduler_dict[model_name]
                scheduler.load_state_dict(state['scheduler'])
            self.start_epoch = state['epoch'] + 1
            self.epoch = state['epoch'] + 1
            del(state)
        try:
            # Update to best metric
            self.best = io.load_json(os.path.join(self.args.experiment.experiment_dir,
                                  'best_{}.json'.format(self.args.train.metric)))[self.args.train.metric]
        except:
            print('[Warning] Cannot load history metric')


    def load_or_not(self, model_names=None):
        # Load weights or not
        if model_names is None:
            model_names = self.model_names
        if self.args.overall.test:
            self.load(flag='best')
        elif self.args.overall.resume < 0:
            self.load(flag='last')
        elif self.args.overall.resume > 0:
            self.load(flag=self.args.overall.resume)
        else:
            pass

    def _update_meta(self, meta):
        # Update history image for smooth learning
        return meta

    '''
       ==================================
       ==== Train & Test & Evaluate =====
       ==================================
    '''
    def train(self, train_dataloader, val_dataloader=None):
        '''
        The core training function
        :param train_dataloader: train data loader
        :param val_dataloader: val data loader
        :return:
        '''
        # Obtain overall parameters
        self.step = train_dataloader.__len__()
        self.val_step = val_dataloader.__len__()
        self.N_batch = self.step * self.args.train.epochs
        # Start to train !
        self.on_train_begin()
        for epoch in range(self.start_epoch, self.args.train.epochs + 1):
            du.shuffle_dataset(train_dataloader, epoch)
            self.on_epoch_begin()
            for i, batch in enumerate(train_dataloader):
                self.on_batch_begin()
                losses, meta = self.train_on_batch(batch)
                losses = self.reduce_item(losses)
                self.on_batch_end(losses, meta)
            if not self.args.train.no_val:
                [metric] = self.test(val_dataloader)
                self.on_epoch_end(metric)
            else:
                self.on_epoch_end()
        self.on_train_end()

    def train_on_batch(self, data) -> dict:
        '''
        Train on the batch
        :param data: input data
        :return: losses dictionary and meta data
        '''
        losses = {}
        losses, meta = self._train_gen(data, losses)
        losses = self.reduce_item(losses)
        meta = self._update_meta(meta)
        if self.args.train.lambda_gan > 0:
            losses = self._train_dis(meta, losses)
        return losses, meta

    def _train_gen(self, losses):
        '''
        Train generator
        :param data: input data
        :param losses: losses dictionary
        :return:
        '''
        pass

    def _train_dis(self, meta, losses):
        '''
        Train discriminator
        :param meta: image data
        :param losses: losses dictionary
        :return:
        '''
        pass

    @torch.no_grad()
    def test(self, val_dataloader, evaluate=True, return_pred=False):
        '''

        :param val_dataloader: the test dataloader
        :param evaluate:  evaluate or not
        :param return_pred: return pred tensors or not
        :return: metrics / tensors
        '''
        self.eval()
        source = []
        target = []
        metric = {}
        for i, batch in enumerate(val_dataloader):
            s, t = self.test_on_batch(batch)
            # To avoid CUDA-memory error
            if return_pred:
                source.append(s.cpu())
                target.append(t.cpu())
            if evaluate:
                [s, t] = self.recover_data([s, t])
                metric = self.evaluate_on_batch(s, t, metric)
        self.activate()
        return_list = []
        if evaluate:
            for key in metric:
                metric[key] = np.mean(metric[key])
            return_list.append(metric)
        if return_pred:
            source = torch.cat(source, dim=0)
            target = torch.cat(target, dim=0)
            return_list += [source, target]
        return return_list

    def eval(self):
        '''
        set networks eval mode
        :return:
        '''
        for model_name in self.model_names:
            getattr(self, model_name).eval()

    def activate(self):
        '''
        set networks trainable
        :return:
        '''
        for model_name in self.model_names:
            getattr(self, model_name).train()

    def test_on_batch(self, batch):
        '''
        Test on batch
        :param batch: The batch data
        :return: the original image and enhanced image
        '''
        if self.pair: 
            real_bad = batch['noise_good'].to(self.args.overall.device)
            rec_good = self.b2g_gen(real_bad).detach()
            gt = batch['good'].to(self.args.overall.device)
            if self.args.dist.n_gpus > 1:
                gt, rec_good = du.all_gather([gt, rec_good])  # TODO: check the GPU-memory
            return rec_good, gt
        else:
            real_bad = batch['bad'].to(self.args.overall.device)
            fake_good = self.b2g_gen(real_bad).detach()
            # Evaluate with real bad
            # It is not reasonable to calculate metric based on low quality image
            if self.args.dist.n_gpus > 1:
                [real_bad, fake_good] = du.all_gather([real_bad, fake_good])  # TODO: check the GPU-memory
            return fake_good, real_bad

    def recover_data(self, data):
        '''
        changed the normalized data into (0, 1)
        :param data: normalized data
        :return: un-normalized data
        '''
        un_norm = UnNormalize(self.args.data.mean, self.args.data.std)
        if isinstance(data, list):
            un_data = []
            for one_data in data:
                un_data.append(torch.cat([un_norm(one_data[k, ...]).unsqueeze(dim=0) for k in range(one_data.size(0))]))
            return un_data
        return torch.cat([un_norm(data[k, ...]).unsqueeze(dim=0) for k in range(data.size(0))], dim=0)

    def evaluate_on_batch(self, source, target, metric=None):
        '''
        :param source: usually enhanced image
        :param target: usually ground-truth image
        :param metric: a dictionary containing keywords of psnr/ssim
        :return:
        '''
        if metric is None:
            metric = {}
        if 'psnr' not in metric.keys():
            metric['psnr'] = []
        metric['psnr'] += PSNR(source, target)
        if 'ssim' not in metric.keys():
            metric['ssim'] = []
        metric['ssim'] += SSIM(source, target)
        return metric

    def evaluate(self, source, target, metric=None) -> dict:
        # for large-scale data evaluation, you need to convert input into cpu
        if metric is None:
            metric = {}
        metric['psnr'] = PSNR(source, target)
        metric['ssim'] = SSIM(source, target)
        return metric







