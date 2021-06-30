from isecret.utils.metric import PSNR, SSIM
import isecret.utils.distributed as du
from isecret.model.common import MyModel
from isecret.model.build import MODEL_REGISTRY
from isecret.model.backbone import BACKBONE_REGISTRY, DISCRIMINATOR_REGISTRY, FMODEL_REGISTRY
from isecret.train_utils.utils import set_requires_grad, ImagePool, print_network
from isecret.loss import ISLoss, ICCLoss, LSGANLoss
from isecret.train_utils.optimizer import make_optimizer
from isecret.train_utils.scheduler import SCHEDULER_REGISTRY
import torch.nn as nn
import numpy as np
import torch
import kornia


@ MODEL_REGISTRY.register('i-secret')
class ISECRET(MyModel):
    def __init__(self, args):
        MyModel.__init__(self, args)
        self.pair = True
    

    def _check_network(self):
        if self.args.train.lambda_is > 0:
            if 'importance' not in self.args.model.generator:
                self.args.model.generator = 'importance-' + self.args.model.generator
        else:
            if self.args.train.lambda_icc > 0:
                raise RuntimeError('ICC loss should be based IS loss, while lambda_is is zero')

    def _modify_args(self, args):
        self.args.train.lambda_gan = 1.0
        self.args.train.lambda_icc = 1.0
        self.args.train.lambda_is = 1.0
        if 'importance' not in self.args.model.generator:
            self.args.model.generator = 'importance-' + self.args.model.generator


    def build_model(self):
        if self.args.overall.test:
            self.model_names = ['b2g_gen']
            self.load_model()
            self._parallel()
        else:
            self.model_names = ['b2g_gen', 'good_dis', 'f_model']
            # Please note that MLP should be initialized at the beginning of the training
            if self.args.overall.resume != 0:
                self.load_model()
                self._parallel()
            else:
                self.b2g_gen = BACKBONE_REGISTRY. \
                    get(self.args.model.generator.lower())(self.args)
                self.good_dis = DISCRIMINATOR_REGISTRY.\
                    get(self.args.model.discriminator.lower())(self.args)
                self.f_model = FMODEL_REGISTRY.\
                    get(self.args.model.f_model.lower())(self.args, torch.cuda.current_device())
                self._parallel(['b2g_gen', 'good_dis'])


    def on_train_begin(self):
        self.build_logger()
        self.good_pool = ImagePool(pool_size=50)
        self.epoch = self.args.train.start_epoch
        if not self.master:
            return
        if 'loss' in self.args.train.metric:
            self.opt = np.less
            self.best = np.inf
        else:
            self.opt = np.greater
            self.best = -np.inf

    def _set_f(self, batch):
        # This function is just used to initialize the f model
        if self.epoch == 1:
            real_good, real_bad = batch['good'].to(self.args.overall.device), batch['bad'].to(self.args.overall.device)
            feat_k, _ = self.b2g_gen(real_bad, layers=self.args.train.nce_layers)
            self.f_model(feat_k, self.args.train.n_patches, None)  # Initialize
            self._parallel(['f_model'])
        self.build_loss()
        self.build_optimizer()
        self.build_scheduler()
        self.load_or_not()
        if self.epoch == 1:
            self.save_model()
        for name in self.model_names:
            print_network(getattr(self, name), name)

    def build_optimizer(self):
        MyModel.build_optimizer(self)
        self.f_optimizer = make_optimizer(self.args, self.f_model.parameters())
        self.optim_dict['f_model'] = self.f_optimizer

    def build_scheduler(self):
        MyModel.build_scheduler(self)
        self.f_scheduler = SCHEDULER_REGISTRY.get(self.args.train.scheduler) \
            (self.f_optimizer, T_max=self.N_batch)
        self.scheduler_dict['f_model'] = self.f_scheduler

    def build_loss(self):
        MyModel.build_loss(self)
        self._nce_losses = []
        for nce_layer in self.args.train.nce_layers:
            self._nce_losses.append(ICCLoss(self.args).to(self.args.overall.device))
        self.icc_loss = self._nce_loss
        self.gan_loss = LSGANLoss(self.args)
        self.idt_loss = self._nce_loss
        self.ssim_loss = kornia.losses.SSIM(window_size=11)  # TODO: Check the windows size
        self.psnr_loss = kornia.losses.PSNRLoss(max_val=1.0)
        if self.args.train.lambda_is > 0:
            self.rec_loss = ISLoss()
        else:
            self.rec_loss = nn.MSELoss()

    def _nce_loss(self, source, target, weight_map=None):
        feat_q, _ = self.b2g_gen(target, layers=self.args.train.nce_layers)
        feat_k, _  = self.b2g_gen(source, layers=self.args.train.nce_layers)
        if weight_map is None:
            feat_k_pool, sample_ids = self.f_model(feat_k, self.args.train.n_patches)
            feat_q_pool, _ = self.f_model(feat_q, self.args.train.n_patches, sample_ids)
            nce_loss = 0.
            for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self._nce_losses, self.args.train.nce_layers):
                nce_loss += crit(f_q, f_k).mean()
        else:
            feat_k_pool, sample_ids, weight_sample = self.f_model(feat_k, self.args.train.n_patches, None, weight_map=weight_map)
            feat_q_pool, _ = self.f_model(feat_q, self.args.train.n_patches, sample_ids)
            nce_loss = 0.
            for f_q, f_k, crit, weight, nce_layer in zip(feat_q_pool, feat_k_pool, self._nce_losses, weight_sample, self.args.train.nce_layers):
                nce_loss += crit(f_q, f_k, weight).mean()
        nce_loss /= len(self.args.train.nce_layers)
        return nce_loss

    def _update_meta(self, meta: dict):
          # Update history image for smooth learning
          # meta['fake_good'] = self.good_pool.query(meta['fake_good'])
          return meta

    def _adjust_lr(self):
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()
            self.dis_scheduler.step()
            self.f_scheduler.step()
    
    def _vis_importance(self, importance):
        importance = torch.exp(importance)
        for idx in range(importance.shape[0]):
            importance[idx, ...] = (importance[idx, ...] - torch.min(importance[idx, ...])) \
            / (torch.max(importance[idx, ...]) - torch.min(importance[idx, ...]))
        return importance

    def train(self, train_dataloader, val_dataloader=None):
        # Over write
        # Obtain overall parameters
        self.step = len(train_dataloader)
        self.val_step = len(val_dataloader)
        self.N_batch = self.step * self.args.train.epochs
        # Start to train !
        first = True
        self.on_train_begin()
        for epoch in range(self.args.train.start_epoch + 1, self.args.train.epochs):
            du.shuffle_dataset(train_dataloader, epoch)
            if not first:
                self.on_epoch_begin()
            for i, batch in enumerate(train_dataloader):
                if first:
                    self._set_f(batch)
                    first = False
                    self.on_epoch_begin()
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

    def _train_gen(self, data, losses):
        set_requires_grad(self.good_dis, False)
        self.gen_optimizer.zero_grad()
        self.f_optimizer.zero_grad()
        meta = {}

        # Data prepare
        meta['real_good'],  meta['real_bad'], meta['noise_good'] = data['good'].to(self.args.overall.device, non_blocking=True), data['bad'].to(self.args.overall.device, non_blocking=True), data['noise_good'].to(self.args.overall.device, non_blocking=True)

        # Supervised training
        losses, meta = self._train_supervised(losses, meta)

        # Unsupervised training is only activated when lambda_icc > 0
        if self.args.train.lambda_icc > 0:
             losses, meta = self._train_unsupervised(losses, meta)

        # Calculate loss and backward
        loss = 0
        for loss_value in losses.values():
            loss += loss_value
        loss.backward()
        self.gen_optimizer.step()
        self.f_optimizer.step()
        set_requires_grad([self.good_dis], True)
        return losses, meta

    
    def _train_supervised(self, losses, meta):
       
        if self.args.train.lambda_is > 0:
            meta['rec_good'], importance_rec = self.b2g_gen(meta['noise_good'], need_importance=True)
            losses['is_loss'] = self.rec_loss(meta['rec_good'], meta['real_good'], importance_rec) * self.args.train.lambda_is
            meta['importance_rec'] = self._vis_importance(importance_rec)
        else:
            meta['rec_good'] = self.b2g_gen(meta['noise_good'])
            losses['rec_loss'] = self.rec_loss(meta['rec_good'], meta['real_good']) * self.args.train.lambda_rec

        # Baseline model only use gan to disciminate the degraded image
        if self.args.train.lambda_gan > 0 and self.args.train.lambda_icc <= 0:
            losses['gan_loss'] = \
                self.gan_loss.update_g(self.good_dis, meta['rec_good']) * self.args.train.lambda_gan

        return losses, meta
    
    def _train_unsupervised(self, losses, meta):

        # Forward network
        real = torch.cat((meta['real_good'], meta['real_bad']), dim=0)
        fake, importance = self.b2g_gen(real, need_importance=True)
        _, importance_fake = importance.chunk(2, dim=0)
        importance_fake = importance_fake.detach()

        # Visualize importance map
        meta['importance_fake'] = self._vis_importance(importance_fake)
        meta['idt_good'], meta['fake_good'] = fake.chunk(2, dim=0)

        # Calculate icc-loss
        if self.args.train.lambda_icc > 0:
            losses['icc_loss'] = self.icc_loss(meta['real_bad'], meta['fake_good'], importance_fake) * self.args.train.lambda_icc

        # Calculate idt-loss
        if self.args.train.lambda_idt > 0:
            losses['idt_loss'] = self.idt_loss(meta['real_good'], meta['idt_good']) * self.args.train.lambda_idt

        # Calculate gan-loss
        if self.args.train.lambda_gan > 0:
            losses['gan_loss'] = \
                self.gan_loss.update_g(self.good_dis, meta['fake_good']) * self.args.train.lambda_gan

        return losses, meta

       
      
    def _train_dis(self, meta, losses):
        set_requires_grad([self.b2g_gen], False)
        self.dis_optimizer.zero_grad()
         # Baseline model only use gan to disciminate the degraded image
        if self.args.train.lambda_gan > 0 and self.args.train.lambda_icc <= 0:
            losses['dis_loss'] = self.gan_loss.update_d(self.good_dis, meta['real_good'],
                                                        meta['rec_good'])
        elif self.args.train.lambda_gan > 0 and self.args.train.lambda_icc > 0:
            losses['dis_loss'] = self.gan_loss.update_d(self.good_dis, meta['real_good'],
                                                        meta['fake_good'])       
        # Backward loss
        loss = losses['dis_loss']
        loss.backward()
        self.dis_optimizer.step()
        set_requires_grad([self.b2g_gen], True)
        return losses

   

