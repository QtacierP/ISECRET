# -*- coding: utf-8 -*-
import isecret.utils.distributed as du
import isecret.utils.loggging as logging
from isecret.model.build import build_model
from isecret.dataset.build import build_loader
from isecret.utils.sender import EMailSender
from isecret.utils.std_logger import StdLog
import numpy as np
import random
import torch
import time
import sys



logger = logging.get_logger(__name__)


def train_func(args):
    # Initialize distributed training
    du.init_distributed_training(args)

    # Fix random seed
    torch.manual_seed(args.overall.seed)
    np.random.seed(args.overall.seed)
    random.seed(args.overall.seed)

    # Set torch flag
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Log training setting
    '''logging.setup_logging(args.experiment.log_dir)
    logger.info("Train with config:\n")
    logger.info(args)'''

    if du.is_master_proc:
        logger = StdLog(args.experiment.current_log_dir)
        sys.stdout = logger
    print(args)

    # Build data loader
    train_dataloader, val_dataloader, test_dataloader = build_loader(args)

    # Build model
    model = build_model(args)

    # train\
    if not args.overall.test:
        model.train(train_dataloader, val_dataloader)

    # evaluate
    # Load model to the best mode
    model.load()
    metric = model.test(test_dataloader)

    if du.is_master_proc():
        print(metric)


   