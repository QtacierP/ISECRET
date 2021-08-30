# -*- coding: utf-8 -*-
import argparse
import os
import yaml
from bunch import Bunch
import time



args = argparse.ArgumentParser(description='the option of the project Fundus-Enhancement')

# Overall option
overall_args = args.add_argument_group(description='overall')
overall_args.add_argument('--seed', type=int, default=0,
                          help='the random seed of the whole project')
overall_args.add_argument('-device', type=str, default='cuda',
                          help='the running device of the whole project')
overall_args.add_argument('--test', action='store_true', default=False, help='test mode or not')
overall_args.add_argument('--test_dir', type=str, default='', help='test dir')
overall_args.add_argument('--gt_dir', type=str, default='', help='gt dir')
overall_args.add_argument('--output_dir', type=str, default='./temp', help='test output dir')
overall_args.add_argument('--resume', type=int, default=0, help='resume training or not')

# Dataset option
data_args = args.add_argument_group(description='data')
data_args.add_argument('--data_root_dir', type=str, default='/raid5/pujin/MedicalEnhancement/data',
                       help='the root directory of the dataset')
data_args.add_argument('--dataset', type=str, default='eyeq',
                       help='the dataset used for training/validation. Usable : crop_eyeq | eyeq')
data_args.add_argument('--size', type=int, default=512, help='the image size of the input data')
data_args.add_argument('--crop_size', type=int, default=-1, help='the cropped size of the input data')



# Distribution training option
dist_args = args.add_argument_group(description='dist')
dist_args.add_argument('--gpu', type=str, default='0', help='the index of the GPU')
dist_args.add_argument('--dist_backend', type=str, default='nccl', help='The backned of the distribution training')
dist_args.add_argument('--init_method', type=str,
                       default='tcp://localhost:10098',
                       help='initialization method for distribution training. Usable: tcp://localhost:xxxx | file://xxx')
dist_args.add_argument('--num_shards', type=int, default=1, help='the number of machines used for training')
dist_args.add_argument('--shard_id', type=int, default=0, help='the index of the current machine')
dist_args.add_argument('--num_worker', type=int, default=0,
                       help='the number of workers for multi-processes dataloader')
dist_args.add_argument('--batch_size', type=int, default=2, help='the batch size for training')

# TODO: we only support single machine currently. A future version with cluster training will be updated
# Experiment option
experiment_args = args.add_argument_group(description='experiment')
experiment_args.add_argument('--name', type=str, default='debug', help='The name of the experiment')
experiment_args.add_argument('--experiment_root_dir', type=str, default='/raid5/pujin/MedicalEnhancement/experiment',
                                   help='The directory of the experiment')


# training option
train_args = args.add_argument_group(description='train')
train_args.add_argument('--default', action='store_true', default=False, help='use default setting or not')
train_args.add_argument('--epochs', type=int, default=200, help='the number of epochs for training')
train_args.add_argument('--len', type=int, default=0, help='the number of data for training, 0 means all')
train_args.add_argument('--save_freq', type=int, default=10, help='the number of epoch for saving model')
train_args.add_argument('--sample_freq', type=int, default=500, help='the number of step for visualizing')
train_args.add_argument('--no_val', action='store_true', default=False, help='no validation used in training')
train_args.add_argument('--metric', type=str, default='psnr', help='the validation metric')
train_args.add_argument('--optim', type=str, default='adam_belief', help='Optimizer')
train_args.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
train_args.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam')
train_args.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optim')
train_args.add_argument('--lr', type=float, default=1e-4, help='learning rate for optim')
train_args.add_argument('--scheduler', type=str, default='cosine', help='scheduler for optimizer')
train_args.add_argument('--nce_layers', type=str, default='1,5,9,11,15,18,20', help='the index of nce layers')
train_args.add_argument('--nce_T', type=float, default=0.07, help='the temperature weight in nce-loss')
train_args.add_argument('--n_patches', type=int, default=256, help='the f_model used in CutGAN')
train_args.add_argument('--lambda_gan', type=float, default=1.0, help='the weight of the gan-loss')
train_args.add_argument('--lambda_icc', type=float, default=0.0, help='the weight of the icc-loss')
train_args.add_argument('--lambda_idt', type=float, default=0.0, help='the weight of the idt-loss')
train_args.add_argument('--lambda_simsiam', type=float, default=0.0, help='the weight of the simsiam-loss')
train_args.add_argument('--lambda_ssim', type=float, default=0.0, help='the weight of the ssim-loss')
train_args.add_argument('--lambda_idt_ssim', type=float, default=0.0, help='the weight of the idt-ssim-loss')
train_args.add_argument('--lambda_psnr', type=float, default=0.0, help='the weight of the psnr-loss')
train_args.add_argument('--lambda_rec', type=float, default=1.0, help='the weight of the rec-loss')
train_args.add_argument('--lambda_is', type=float, default=0.0, help='the weight of the is-loss')



# model option
model_args = args.add_argument_group(description='model')
model_args.add_argument('--model', type=str, default='i-secret', help='model. Usable: i-secret')
model_args.add_argument('--n_layers', type=int, default=3, help='the number of layers used in discriminator')
model_args.add_argument('--n_downs', type=int, default=2, help='the number of down-samplings used in generator')
model_args.add_argument('--n_filters', type=int, default=64, help='the number of filters used in generator')
model_args.add_argument('--n_blocks', type=int, default=9,
                        help='the number of blocks used in the low-level of generator')
model_args.add_argument('--input_nc', type=int, default=3, help='the number of input channels')
model_args.add_argument('--output_nc', type=int, default=3, help='the number of output channels')
model_args.add_argument('--dis_n_filters', type=int, default=64, help='the number of filters used in discriminator')
model_args.add_argument('--norm', type=str, default='in', help='Norm layer. Usable: in | bn ')
model_args.add_argument('--padding', type=str, default='reflect', help='padding layer. '
                                                                       'Usable: reflect | replication | zero ')
model_args.add_argument('--gan_loss', type=str, default='lsgan', help='GAN loss. Usable: lsgan ')
model_args.add_argument('--generator', type=str,
                        default='resnet', help='Generator backbone. Usable:  resnet ')
model_args.add_argument('--discriminator', type=str,
                        default='patchgan', help='Discriminator . Usable: patchgan ')
model_args.add_argument('--f_model', type=str, default='mlp', help='the f_model used in CutGAN')
model_args.add_argument('--no_mlp', action='store_true', default=False, help='do not use mlp in CutGAN')






args = args.parse_args()


def convert_config(config):
    config = Bunch(config)  # convert dictionary into class
    for key_word in config.keys():
        setattr(config, key_word, Bunch(getattr(config, key_word)))
    return config


def post_preprocess_args(args, verbose=False):
    args, args_dict = arrange_args(args)
    args = init_dir(args)
    args, args_dict = update_args(args, args_dict, verbose)
    args = init_dir(args)
    args.dist.n_gpus = len(args.dist.gpu.split(','))
    args.train.nce_layers = [int(i) for i in args.train.nce_layers.split(',')]
    args.dist.batch_size_per_gpu = args.dist.batch_size // args.dist.n_gpus
    return args, args_dict


def arrange_args(args):
    # Re-arrange the name space
    args_dict = {}
    sub_args_list = [data_args, dist_args, experiment_args, train_args, model_args, overall_args]
    for sub_args in sub_args_list:
        sub_options = [action.dest for action in sub_args._group_actions]
        sub_names = {name: value for (name, value) in args._get_kwargs() if name in sub_options}
        sub_node = {}
        for sub_name in sub_names:
            sub_node[sub_name] = sub_names[sub_name]
        args_dict[sub_args.description] = sub_node
    args = convert_config(args_dict)
    return args, args_dict


def init_dir(args, verbose=False):
    # Initialize the directory structure in args
    args.data.data_dir = os.path.join(args.data.data_root_dir, args.data.dataset)
    args.experiment.experiment_dir = os.path.join(args.experiment.experiment_root_dir,
                                                  args.data.dataset, str(args.data.size),
                                                  args.experiment.name)

    args.experiment.ckpt_dir = os.path.join(args.experiment.experiment_root_dir,
                                            args.data.dataset, str(args.data.size),
                                            args.experiment.name, 'checkpoint')
    args.experiment.config_dir = os.path.join(args.experiment.experiment_root_dir,
                                               args.data.dataset, str(args.data.size),
                                               args.experiment.name, 'config')
    args.experiment.log_dir = os.path.join(args.experiment.experiment_root_dir,
                                           args.data.dataset, str(args.data.size),
                                           args.experiment.name, 'log')
    args.experiment.current_log_dir = os.path.join(args.experiment.log_dir, '{}.log'.format(time.time()))
    if not os.path.exists(args.experiment.ckpt_dir):
        os.makedirs(args.experiment.ckpt_dir)
        print('[INFO] making experiment directory at {}...'.format(args.experiment.experiment_dir))
    if not os.path.exists(args.experiment.log_dir):
        os.makedirs(args.experiment.log_dir)
    return args



def update_args(args, args_dict, verbose=False):
    # Update args via testing/resume or training
    if args.overall.test:
        # Load anything except dist_args
        load_args, load_args_dict = yaml2args(args.experiment.config_dir)
        update_list = ['dist', 'overall', 'experiment']
        for update_attribution in update_list:
            setattr(load_args, update_attribution, getattr(args, update_attribution))
            load_args_dict[update_attribution] = args_dict[update_attribution]
    elif args.overall.resume != 0:
        # Resume
        load_args, load_args_dict = yaml2args(args.experiment.config_dir)
        update_list = ['dist', 'overall', 'experiment']
        for update_attribution in update_list:
            setattr(load_args, update_attribution, getattr(args, update_attribution))
            load_args_dict[update_attribution] = args_dict[update_attribution]
        # Maintain the resume flag
        load_args.train.start_epoch = args.overall.resume + 1
        # Load lambda_adjust
        try: # Avoid version conflict
            if load_args.train.lambda_adjust != '':
                lambda_current = getattr(load_args.train, load_args.train.lambda_adjust)
                lambda_loss = lambda_current + (load_args.train.start_epoch) / load_args.train.epochs * \
                              (load_args.args.train.lambda_target - lambda_current)
        except:
            pass
    else:
        if not verbose:
            # Train. We need to avoid the overwrite problem
            if os.path.exists(args.experiment.config_dir):
                response = input('[Warning] : An old yaml has existed.'
                                 ' Do you want to overwrite it ? [y/n] \n')
                while response != 'n' and response != 'y':
                    response = input('Please input y or n \n')
                if response == 'n':
                    print('[INFO] : Program will exit, '
                          'and please rename your experiment and run again')
                    exit(0)
            else:
                os.makedirs(args.experiment.config_dir)
        args2yaml(args_dict, args.experiment.config_dir)
        load_args, load_args_dict = args, args_dict
        load_args.train.start_epoch = 1

    # Assign differnt std/mean to differnt value
    if 'eyeq' in args.data.dataset.lower():
        load_args.data.mean = [0.5, 0.5, 0.5]
        load_args.data.std = [0.5, 0.5, 0.5]
    else:
        load_args.data.mean = [0.5, 0.5, 0.5]
        load_args.data.std = [0.5, 0.5, 0.5]

    # Assign patches to dataset
    if load_args.data.crop_size == -1:
        load_args.data.crop_size = load_args.data.size
    n = load_args.data.size // load_args.data.crop_size
    if n > 2:
        print('[Info] Use patch mode, each image is divided into {} x {}'.format(n, n))
    load_args.train.true_epochs = n * n * load_args.train.epochs
    load_args.train.step = n * n

    load_args.data.dataformat = 'CHW' if load_args.model.input_nc == 3 else 'HW'
    return load_args, load_args_dict


def args2yaml(args_dict, yaml_dir):
    # Save args to .yaml
    f = open(os.path.join(yaml_dir, 'config.yaml'), 'w')
    yaml.dump(args_dict, f)
    f.close()


def yaml2args(yaml_dir):
    # Load .yaml to args
    f = open(os.path.join(yaml_dir, 'config.yaml'), 'r')
    args_dict = yaml.load(f)
    f.close()
    loaded_args = convert_config(args_dict)
    return loaded_args, args_dict





