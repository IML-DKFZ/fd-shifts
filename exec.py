


import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import utils.exp_utils as utils
import utils.eval_utils as eval_utils
import utils.model_utils as model_utils
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import default_args as parsed_args
from tqdm import tqdm
from collections import OrderedDict
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import TensorBoardLogger

def train():
    """
    perform the training routine for a given fold. saves plots and selected parameters to the experiment dir
    specified in the configs.
    """
    data_loader = utils.import_module('dl', os.path.join(args.exec_dir, args.dataset_source, 'svhn_loader.py'))
    datamodule = data_loader.DataLoader(args)
    tb_logger= TensorBoardLogger(save_dir=args.exp_dir)
    csv_logger= CSVLogger(save_dir=args.exp_dir)
    trainer = pl.Trainer(gpus=1, logger=[tb_logger, csv_logger])
    model = utils.import_module('model', args.model_path).net(args).cuda()
    trainer.fit(model=model, datamodule=datamodule)


def test():

    results_dict = eval_utils.test_sym2img(args, logger)
    utils.write_test(args, writer, results_dict)
    writer.close()


if __name__ == '__main__':


    in_args = parsed_args.Args()
    folds = in_args.folds
    if in_args.exec_dir is None:
        in_args.exec_dir = os.path.dirname(os.path.realpath(__file__))  # current dir.

    if in_args.mode == 'train' or in_args.mode == 'train_test':

        args = utils.prep_exp(in_args, use_stored_settings=in_args.use_stored_settings)
        if folds is None:
            folds = range(args.n_cv_splits)

        for fold in folds:
            args.fold_dir = os.path.join(args.exp_dir, 'fold_{}'.format(fold))
            args.fold = fold
            train()
            if in_args.mode == 'train_test':
                test()

    elif in_args.mode == 'test':

        args = utils.prep_exp(in_args, is_training=False, use_stored_settings=True)
        writer = SummaryWriter(log_dir=args.plot_dir)
        if in_args.apply_test_config_mods:
            args = utils.apply_test_config_mods(args)
        data_loader = utils.import_module('dl', os.path.join(args.exec_dir, args.dataset_source, 'svhn_loader.py'))
        if folds is None:
            folds = range(args.n_cv_splits)

        for fold in folds:
            args.fold_dir = os.path.join(args.exp_dir, 'fold_{}'.format(fold))
            args.checkpoint_path = os.path.join(args.fold_dir, args.checkpoint_name)
            logger = utils.get_logger(args.fold_dir)
            args.fold = fold
            test()

    else:
        raise RuntimeError('mode specified in in_args is not implemented...')

    utils.write_halt(args, logger, writer)

