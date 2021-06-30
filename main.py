# -*- coding: utf-8 -*-
from option import args, post_preprocess_args
from isecret.utils.launch import launch_job
from isecret.utils.std_logger import StdLog
from train import train_func
from test import test_func
import sys


# Set warning information invisible
import warnings
warnings.filterwarnings("ignore")


'''if torch.distributed.get_rank() > 0:
    fitlog.debug()

fitlog.commit(__file__)             # auto commit your codes
fitlog.add_hyper_in_file(__file__)  # record your hyper-parameters'''

if __name__ == '__main__':
    # Fix random seed

    args, args_dict = post_preprocess_args(args)
    print('[INFO] GPU Index ', args.dist.gpu)
    if not args.overall.test:
        launch_job(args, train_func)
    else:
        launch_job(args, test_func)

#fitlog.finish()
