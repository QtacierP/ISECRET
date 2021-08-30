# -*- coding: utf-8 -*-
from torch.nn.modules.pooling import AdaptiveAvgPool2d
from isecret.utils.registry import Registry
import torch.nn as nn
from isecret.model.utils import make_norm, make_paddding, check_architecture
from isecret.model.blocks import ResnetBlock
import torch
import functools
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import os

BACKBONE_REGISTRY = Registry('BACKBONE')
BACKBONE_REGISTRY.__doc__ = 'Backbone for building model'
DISCRIMINATOR_REGISTRY = Registry('DISCRIMINATOR')
DISCRIMINATOR_REGISTRY.__doc__ = 'Discriminator for building model'
FMODEL_REGISTRY = Registry('F MODEL')
FMODEL_REGISTRY.__doc__ = 'F model for CutGAN'

@ BACKBONE_REGISTRY.register('resnet')
class ResGenerator(nn.Module):
    def __init__(self, args, input_nc=None, output_nc=None):
        nn.Module.__init__(self)
        self.args = args

        # Judge fundamental setting
        check_architecture(args)

        # Get padding
        padding = make_paddding(args)

        # Get norm
        norm_layer, use_bias = make_norm(args)
        if input_nc is None:
            input_nc = self.args.model.input_nc
        if output_nc is None:
            output_nc = self.args.model.output_nc

        # Build Head
        model = [padding(3),
                nn.Conv2d(input_nc, self.args.model.n_filters, kernel_size=7, bias=use_bias),
                norm_layer(self.args.model.n_filters),
                nn.ReLU(True)]

        # Build down-sampling
        for i in range(self.args.model.n_downs):
            mult = 2 ** i
            model += [padding(1), nn.Conv2d(self.args.model.n_filters * mult,
                                            self.args.model.n_filters * mult * 2,
                                            kernel_size=3, stride=2, bias=use_bias),
                      norm_layer(self.args.model.n_filters * mult * 2),
                      nn.ReLU(True)]
        mult = 2 ** self.args.model.n_downs

        # Build rev-blocks
        self.in_ch = self.args.model.n_filters * mult * 4
        for i in range(self.args.model.n_blocks):
            model += [ResnetBlock(self.args.model.n_filters * mult, padding=padding,
                                norm_layer=norm_layer, use_bias=use_bias)]

        # Build up-sampling
        for i in range(self.args.model.n_downs):
            mult = 2 ** (self.args.model.n_downs - i)
            model += [nn.ConvTranspose2d(self.args.model.n_filters * mult,
                                      int(self.args.model.n_filters * mult / 2),
                                      kernel_size=3, stride=2,
                                      padding=1, output_padding=1,
                                      bias=use_bias),
                   norm_layer(int(self.args.model.n_filters * mult / 2)),
                   nn.ReLU(True)]

        # Build tail
        model += [padding(3)]
        model += [nn.Conv2d(self.args.model.n_filters, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        # Make model
        self.model = nn.Sequential(*model)

    def forward(self, input, layers=None):
        if layers is None:
            x = self.model(input)
            return x
        else:
            return self.forward_features(input, layers)

    def forward_features(self, input, layers):
        feat = input
        feats = []
        for layer_id, layer in enumerate(self.model):
            feat = layer(feat)
            if layer_id in layers:
                feats.append(feat)
        return feats, feat


@ BACKBONE_REGISTRY.register('importance-resnet')
class ImportanceResGenerator(nn.Module):
    def __init__(self, args):
        nn.Module.__init__(self)
        self.args = args

        # Judge fundamental setting
        check_architecture(args)

        # Get padding
        padding = make_paddding(args)

        # Get norm
        norm_layer, use_bias = make_norm(args)

        # Build Head
        head = [padding(3),
                nn.Conv2d(self.args.model.input_nc, self.args.model.n_filters, kernel_size=7, bias=use_bias),
                norm_layer(self.args.model.n_filters),
                nn.ReLU(True)]

        # Build down-sampling
        downs = []
        for i in range(self.args.model.n_downs):
            mult = 2 ** i
            downs += [padding(1), nn.Conv2d(self.args.model.n_filters * mult,
                                            self.args.model.n_filters * mult * 2,
                                            kernel_size=3, stride=2, bias=use_bias),
                      norm_layer(self.args.model.n_filters * mult * 2),
                      nn.ReLU(True)]
        mult = 2 ** self.args.model.n_downs

        neck = []
        # Build res-blocks
        self.in_ch = self.args.model.n_filters * mult * 4
        for i in range(self.args.model.n_blocks):
            neck += [ResnetBlock(self.args.model.n_filters * mult, padding=padding,
                                norm_layer=norm_layer, use_dropout=False,
                                use_bias=use_bias)]

        ups = []
        # Build up-sampling
        for i in range(self.args.model.n_downs):
            mult = 2 ** (self.args.model.n_downs - i)
            ups += [nn.ConvTranspose2d(self.args.model.n_filters * mult,
                                      int(self.args.model.n_filters * mult / 2),
                                      kernel_size=3, stride=2,
                                      padding=1, output_padding=1,
                                      bias=use_bias),
                   norm_layer(int(self.args.model.n_filters * mult / 2)),
                   nn.ReLU(True)]



        importance_ups = []
        # Build unctainty-aware up-sampling
        for i in range(self.args.model.n_downs):
            mult = 2 ** (self.args.model.n_downs - i)
            importance_ups += [nn.ConvTranspose2d(self.args.model.n_filters * mult,
                                      int(self.args.model.n_filters * mult / 2),
                                      kernel_size=3, stride=2,
                                      padding=1, output_padding=1,
                                      bias=use_bias),
                   norm_layer(int(self.args.model.n_filters * mult / 2)),
                   nn.ReLU(True)]

        
        # Build tail
        ups += [padding(3)]
        ups += [nn.Conv2d(self.args.model.n_filters, self.args.model.output_nc, kernel_size=7, padding=0)]

        ups += [nn.Tanh()]

        # Build importance tail
        importance_ups += [padding(3)]
        importance_ups += [nn.Conv2d(self.args.model.n_filters, self.args.model.output_nc, kernel_size=7, padding=0)]

        # Make model
        self.head = nn.Sequential(*head)
        self.downs = nn.Sequential(*downs)
        self.neck = nn.Sequential(*neck)
        self.ups = nn.Sequential(*ups)
        self.importance_ups = nn.Sequential(*importance_ups)

    def forward(self, input, need_importance=False, layers=None):
        if layers is None:
            x = self.head(input)
            x = self.downs(x)
            x = self.neck(x)
            output = self.ups(x)
            if need_importance:
                importance = self.importance_ups(x)
                return output, importance
            else:
                return output
        else:
            return self.forward_features(input, layers)

    def forward_features(self, input, layers):
        # We only focus on the encoding part
        feat = input
        feats = []
        layer_id = 0
        for layer in self.head:
            feat = layer(feat)
            if layer_id in layers:
                feats.append(feat)
            layer_id += 1
        for layer in self.downs:
            feat = layer(feat)
            if layer_id in layers:
                feats.append(feat)
            layer_id += 1
        for layer in self.neck:
            feat = layer(feat)
            if layer_id in layers:
                feats.append(feat)
            layer_id += 1
        return feats, feat
    

@ DISCRIMINATOR_REGISTRY.register('patchgan')
class PatchDiscriminator(nn.Module):
    def __init__(self, args, input_nc = None):
        nn.Module.__init__(self)
        self.args = args

        # Get norm
        norm_layer, use_bias = make_norm(args)

        if input_nc is None:
            input_nc = self.args.model.input_nc

        # Down-sampling
        model = [nn.Conv2d(input_nc, self.args.model.dis_n_filters,
                           kernel_size=4, stride=2, padding=1, bias=use_bias),
                 nn.LeakyReLU(0.2, True)]

        mult = 1

        for i in range(1, self.args.model.n_layers):
            last_mult = mult
            mult = 2 ** i
            model += [nn.Conv2d(self.args.model.dis_n_filters * last_mult,
                                self.args.model.dis_n_filters * mult,
                                kernel_size=4, stride=2, padding=1, bias=use_bias),
                      norm_layer(self.args.model.dis_n_filters * mult),
                      nn.LeakyReLU(0.2, True)]

        last_mult = mult
        mult = 2 ** self.args.model.n_layers
        model += [nn.Conv2d(self.args.model.dis_n_filters * last_mult,
                            self.args.model.dis_n_filters * mult,
                            kernel_size=4, stride=2, padding=1, bias=use_bias),
                  norm_layer(self.args.model.dis_n_filters * mult),
                  nn.LeakyReLU(0.2, True)]

        model += [nn.Conv2d(self.args.model.dis_n_filters * mult,
                            1, kernel_size=4, stride=2, padding=1, bias=use_bias)]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


'''
========================================
====== Backbone for CutGAN f_model =====
========================================
'''


def init_weights(net, init_type='normal', init_gain=0.02, debug=False):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                print(classname)
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, debug=False, initialize_weights=True):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
    Return an initialized network.
    """
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

class Standardlize(nn.Module):
    def __init__(self):
        super(Standardlize, self).__init__()

    def forward(self, x):
        return x / x.mean()

@ FMODEL_REGISTRY.register('mlp')
class PatchSampleF(nn.Module):
    def __init__(self, args, gpu_id, init_type='xavier', init_gain=0.02, nc=256):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        nn.Module.__init__(self)
        self.args = args
        self.l2norm = Normalize(2)
        self.standard = Standardlize() # Stable gradient 
        self.use_mlp = not self.args.model.no_mlp
        self.nc = nc
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain


    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            feat = feat.cpu()
            input_nc = feat.shape[1]
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None, weight_map=None):
        return_ids = []
        return_feats = []
        return_weight_samples = []
        weight_sample = None
        if self.use_mlp and not self.mlp_init:
            print('[INFO] Create MLP...')
            self.create_mlp(feats)
            self.mlp_init = True
            return
        if weight_map is not None:
            weight_map = weight_map.mean(dim=[1], keepdim=False).unsqueeze(dim=1)
            weight_map = self.standard(torch.exp(-weight_map))
        for feat_id, feat in enumerate(feats):
            B, C, H, W = feat.shape[0], feat.shape[1],  feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
            else:
                x_sample = feat_reshape
                patch_id = []
            if weight_map is not None:
                weight_map = F.interpolate(weight_map, size=(W, H), mode='area')         
                weight_map_reshape = weight_map.permute(0, 2, 3, 1).flatten(1, 2)
                weight_sample = weight_map_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])

            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)
            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
                if weight_map is not None:
                    weight_sample = weight_sample.permute(0, 2, 1).reshape([B, weight_sample.shape[-1], H, W])
            return_feats.append(x_sample)
            if weight_map is not None:
                return_weight_samples.append(weight_sample)
        if weight_map is not None:
            return return_feats, return_ids, return_weight_samples
        return return_feats, return_ids

@ FMODEL_REGISTRY.register('mlp-s')
class PatchSampleS(nn.Module):
    def __init__(self, args, gpu_id, init_type='xavier', init_gain=0.02, nc=256):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        nn.Module.__init__(self)
        self.args = args
        self.l2norm = Normalize(2)
        self.standard = Standardlize()
        self.use_mlp = not self.args.model.no_mlp
        self.nc = nc
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            feat = feat.cpu()
            input_nc = feat.shape[1]
            mlp = nn.Sequential(
                *[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, input_nc)])
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None, weight_map=None):
        return_ids = []
        return_feats = []
        return_p_feats = []
        return_weight_samples = []
        weight_sample = None
        if self.use_mlp and not self.mlp_init:
            print('[INFO] Create MLP...')
            self.create_mlp(feats)
            self.mlp_init = True
            return
        for feat_id, feat in enumerate(feats):
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    patch_id = np.random.permutation(
                        feat_reshape.shape[1])
                    # .to(patch_ids.device)
                    patch_id = patch_id[:int(
                        min(num_patches, patch_id.shape[0]))]
                x_sample = feat_reshape[:, patch_id, :].flatten(
                    0, 1)  # reshape(-1, x.shape[1])
            else:
                x_sample = feat_reshape
                patch_id = []
            if weight_map is not None:
                weight_map = F.interpolate(
                    weight_map, size=(W, H), mode='area')
                weight_map_reshape = weight_map.permute(
                    0, 2, 3, 1).flatten(1, 2)
                weight_map_reshape = self.standard(
                    torch.exp(weight_map_reshape))
                weight_sample = weight_map_reshape[:, patch_id, :].flatten(
                    0, 1)  # reshape(-1, x.shape[1])
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                p_x_sample = mlp(x_sample)

            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)
            p_x_sample = self.l2norm(p_x_sample)
            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape(
                    [B, x_sample.shape[-1], H, W])
                p_x_sample = p_x_sample.permute(0, 2, 1).reshape(
                    [B, p_x_sample.shape[-1], H, W])
                weight_sample = weight_sample.permute(0, 2, 1).reshape(
                    [B, weight_sample.shape[-1], H, W])
            return_feats.append(x_sample)
            return_p_feats.append(p_x_sample)
            if weight_map is not None:
                return_weight_samples.append(weight_sample)
        if weight_map is not None:
            return return_feats, return_p_feats, return_ids, return_weight_samples
        return return_feats, return_p_feats, return_ids